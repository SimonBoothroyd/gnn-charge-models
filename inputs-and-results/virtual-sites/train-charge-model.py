import copy
import functools
import hashlib
import json
import os.path
import pickle
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, cast

import click
import numpy
import rich
import torch
from click_option_group import optgroup
from matplotlib import pyplot
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGeometryKey,
    VirtualSiteParameterType,
)
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm
from openff.recharge.utilities.tensors import to_torch
from openff.toolkit.topology import Molecule
from pydantic import parse_file_as
from pydantic.json import pydantic_encoder
from rich.padding import Padding
from rich.progress import track

_CACHED_CHARGES = {}


def compute_base_charge(
    tagged_smiles,
    conformer_settings: ConformerSettings,
    charge_settings: QCChargeSettings,
):

    if tagged_smiles in _CACHED_CHARGES:
        return _CACHED_CHARGES[tagged_smiles]

    molecule = Molecule.from_mapped_smiles(tagged_smiles, allow_undefined_stereo=True)

    conformers = ConformerGenerator.generate(molecule, conformer_settings)
    charges = QCChargeGenerator.generate(molecule, conformers, charge_settings)

    charge_collection = LibraryChargeCollection(
        parameters=[
            LibraryChargeParameter(
                smiles=tagged_smiles,
                value=[float(v) for v in charges.flatten().tolist()],
            )
        ]
    )
    _CACHED_CHARGES[tagged_smiles] = charge_collection
    return charge_collection


def generate_objective_term(
    esp_record: MoleculeESPRecord,
    conformer_settings: ConformerSettings,
    charge_settings: QCChargeSettings,
    bcc_collection: BCCCollection,
    bcc_parameter_keys: List[str],
    vsite_collection: VirtualSiteCollection,
    vsite_charge_parameter_keys: List[VirtualSiteChargeKey],
    vsite_coordinate_parameter_keys: List[VirtualSiteGeometryKey],
) -> ESPObjectiveTerm:

    with capture_toolkit_warnings():

        objective_term_generator = ESPObjective.compute_objective_terms(
            esp_records=[esp_record],
            charge_collection=compute_base_charge(
                esp_record.tagged_smiles, conformer_settings, charge_settings
            ),
            charge_parameter_keys=[],
            bcc_collection=bcc_collection,
            bcc_parameter_keys=bcc_parameter_keys,
            vsite_collection=(
                None if len(vsite_collection.parameters) == 0 else vsite_collection
            ),
            vsite_charge_parameter_keys=(
                None
                if len(vsite_charge_parameter_keys) == 0
                else vsite_charge_parameter_keys
            ),
            vsite_coordinate_parameter_keys=(
                None
                if len(vsite_coordinate_parameter_keys) == 0
                else vsite_coordinate_parameter_keys
            ),
        )
        return cast(ESPObjectiveTerm, list(objective_term_generator)[0])


def generate_objective_terms(
    esp_records: List[MoleculeESPRecord],
    conformer_settings: ConformerSettings,
    charge_settings: QCChargeSettings,
    bcc_collection: BCCCollection,
    bcc_parameter_keys: List[str],
    vsite_collection: VirtualSiteCollection,
    vsite_charge_parameter_keys: List[VirtualSiteChargeKey],
    vsite_coordinate_parameter_keys: List[VirtualSiteGeometryKey],
    cache_directory: str,
    n_processes: int,
):

    console = rich.get_console()

    def numpy_encoder(obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return pydantic_encoder(obj)

    with console.status("hashing objective term inputs"):

        cache_hash_input = json.dumps(
            [
                [esp_record.json(encoder=numpy_encoder) for esp_record in esp_records],
                conformer_settings.json(),
                charge_settings.json(),
                bcc_collection.json(),
                bcc_parameter_keys,
                vsite_collection.json(),
                vsite_charge_parameter_keys,
                vsite_coordinate_parameter_keys,
            ],
            sort_keys=True,
        )

    cache_hash = hashlib.sha256(cache_hash_input.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_directory, f"train-objective-{cache_hash}.pkl")

    if os.path.isfile(cache_path):

        with console.status("loading cached objective terms"):

            with open(cache_path, "rb") as file:
                objective_terms = pickle.load(file)
                console.print(f"loaded cached objective terms from {cache_path}")
                return objective_terms

    with Pool(processes=n_processes) as pool:

        objective_terms = list(
            track(
                pool.imap(
                    functools.partial(
                        generate_objective_term,
                        conformer_settings=conformer_settings,
                        charge_settings=charge_settings,
                        bcc_collection=bcc_collection,
                        bcc_parameter_keys=bcc_parameter_keys,
                        vsite_collection=vsite_collection,
                        vsite_charge_parameter_keys=vsite_charge_parameter_keys,
                        vsite_coordinate_parameter_keys=vsite_coordinate_parameter_keys,
                    ),
                    esp_records,
                ),
                total=len(esp_records),
                description="building objective terms",
                transient=True,
            )
        )

    with open(cache_path, "wb") as file:
        pickle.dump(objective_terms, file)

    return objective_terms


def vectorize_collections(
    bcc_collection,
    bcc_parameter_keys,
    vsite_collection,
    vsite_charge_parameter_keys,
    vsite_coordinate_parameter_keys,
):
    if len(bcc_parameter_keys) > 0:
        current_charge_increments = bcc_collection.vectorize(bcc_parameter_keys)
    else:
        current_charge_increments = numpy.zeros((0, 1))

    if len(vsite_charge_parameter_keys) > 0:
        current_charge_increments = numpy.vstack(
            [
                current_charge_increments,
                vsite_collection.vectorize_charge_increments(
                    vsite_charge_parameter_keys
                ),
            ]
        )
    current_charge_increments = to_torch(current_charge_increments)
    current_charge_increments.requires_grad = True
    parameters_to_optimize = [current_charge_increments]
    if len(vsite_coordinate_parameter_keys) > 0:
        current_vsite_coordinates = vsite_collection.vectorize_coordinates(
            vsite_coordinate_parameter_keys
        )
        current_vsite_coordinates = to_torch(current_vsite_coordinates)
        current_vsite_coordinates.requires_grad = True

        parameters_to_optimize.append(current_vsite_coordinates)
    else:
        current_vsite_coordinates = None
    return current_charge_increments, current_vsite_coordinates, parameters_to_optimize


def compute_vsite_pentalty(
    values: torch.Tensor, settings: List[Tuple[float, float, float]]
) -> torch.Tensor:

    penalty = torch.zeros(1)

    for value, (element_radii, restraint_strength, restraint_width) in zip(
        values, settings
    ):

        distance = torch.sqrt((value - element_radii * 0.5) ** 2)

        penalty += (
            0.5
            * restraint_strength
            * (distance - restraint_width * 0.5) ** 2
            * torch.where(distance - restraint_width * 0.5 > 0, 1.0, 0.0)
        )

    return penalty.reshape([])


def copy_final_values(
    current_charge_increments: torch.Tensor,
    current_vsite_coordinates: Optional[torch.Tensor],
    bcc_collection,
    bcc_parameter_keys,
    vsite_collection,
    vsite_charge_parameter_keys,
    vsite_coordinate_parameter_keys,
) -> Tuple[BCCCollection, VirtualSiteCollection]:

    bcc_collection = copy.deepcopy(bcc_collection)

    vsite_collection = copy.deepcopy(vsite_collection)
    vsite_collection.aromaticity_model = "MDL"

    final_charge_increments = current_charge_increments.detach().numpy()

    bcc_parameters_by_key: Dict[str, BCCParameter] = {
        bcc.smirks: bcc for bcc in bcc_collection.parameters
    }
    vsite_parameters_by_key: Dict[Tuple[str, str, str], VirtualSiteParameterType] = {
        (vsite.smirks, vsite.type, vsite.name): vsite
        for vsite in vsite_collection.parameters
    }

    for value, key in zip(
        final_charge_increments[: len(bcc_parameter_keys)], bcc_parameter_keys
    ):
        bcc_parameters_by_key[key].value = float(value)

    if len(vsite_charge_parameter_keys) > 0:

        for value, (vsite_smirks, vsite_type, vsite_name, charge_index) in zip(
            final_charge_increments[len(bcc_parameter_keys) :],
            vsite_charge_parameter_keys,
        ):
            vsite_parameter = vsite_parameters_by_key[
                (vsite_smirks, vsite_type, vsite_name)
            ]

            charge_increments = [*vsite_parameter.charge_increments]
            charge_increments[charge_index] = float(value)

            vsite_parameter.charge_increments = tuple(charge_increments)

    if len(vsite_coordinate_parameter_keys) > 0:

        final_vsite_coordinates = current_vsite_coordinates.detach().numpy()

        for value, (vsite_smirks, vsite_type, vsite_name, vsite_attr) in zip(
            final_vsite_coordinates, vsite_coordinate_parameter_keys
        ):
            vsite_parameter = vsite_parameters_by_key[
                (vsite_smirks, vsite_type, vsite_name)
            ]
            setattr(vsite_parameter, vsite_attr, float(value))

    return bcc_collection, vsite_collection


@click.command()
@click.option(
    "--input-esp-records",
    "input_record_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-parameter-coverage",
    "input_coverage_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-parameters",
    "input_parameter_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
)
@optgroup("Parameters")
@optgroup.option(
    "--train-vsite-charge",
    "vsite_charge_parameter_keys",
    type=(str, str, str, int),
    multiple=True,
)
@optgroup.option(
    "--train-vsite-coord",
    "vsite_coordinate_parameter_settings",
    type=(str, str, str, str, float, float, float),
    multiple=True,
)
@optgroup("Hyper-parameters")
@optgroup.option(
    "--learning-rate",
    type=float,
    default=5.0e-3,
    show_default=True,
)
@optgroup.option(
    "--n-epochs",
    type=int,
    default=200,
    show_default=True,
)
@optgroup("Data processing")
@optgroup.option(
    "--n-loader-processes",
    "n_processes",
    type=int,
    default=1,
    show_default=True,
)
def main(
    input_record_path,
    input_coverage_path,
    input_parameter_directory,
    output_directory,
    vsite_charge_parameter_keys,
    vsite_coordinate_parameter_settings,
    learning_rate,
    n_epochs,
    n_processes,
):

    console = rich.get_console()

    console.print("")
    console.rule("train data")
    console.print("")

    with open(input_record_path, "rb") as file:
        with console.status("loading train data"):
            esp_records_train = pickle.load(file)
        console.print(f"loaded {len(esp_records_train)} training records")

    console.print("")
    console.rule("model parameters")
    console.print("")

    conformer_settings, charge_settings = parse_file_as(
        Tuple[ConformerSettings, QCChargeSettings],
        os.path.join(input_parameter_directory, "initial-parameters-base.json"),
    )
    bcc_collection = BCCCollection.parse_file(
        os.path.join(input_parameter_directory, "initial-parameters-bcc.json")
    )
    vsite_collection = VirtualSiteCollection.parse_file(
        os.path.join(input_parameter_directory, "initial-parameters-v-site.json")
    )

    # Determine which parameters will be trained
    with open(input_coverage_path, "r") as file:
        coverage_train = json.load(file)

    vsite_coordinate_parameter_settings = {
        vsite_tuple[:4]: vsite_tuple[4:]
        for vsite_tuple in vsite_coordinate_parameter_settings
    }
    if len(vsite_coordinate_parameter_settings) > 0:
        (vsite_coordinate_parameter_keys, vsite_coordinate_restraint_settings) = zip(
            *vsite_coordinate_parameter_settings.items()
        )
    else:
        (vsite_coordinate_parameter_keys, vsite_coordinate_restraint_settings) = [], []

    assert all(
        key[0] in coverage_train["v-site"] for key in vsite_charge_parameter_keys
    )
    assert all(
        key[0] in coverage_train["v-site"] for key in vsite_coordinate_parameter_keys
    )

    bcc_parameter_keys = [
        parameter.smirks
        for parameter in bcc_collection.parameters
        if coverage_train["bcc"].get(parameter.smirks, 0) > 0
        and parameter.provenance["code"][:2] != parameter.provenance["code"][-2:]
    ]

    console.print("* BCC parameters")
    console.print(
        Padding(
            f"- {len(bcc_parameter_keys)} will be trained and "
            f"{len(bcc_collection.parameters) - len(bcc_parameter_keys)} will be fixed",
            (0, 0, 0, 4),
        )
    )
    console.print("")
    console.print("* v-site parameters")
    console.print(
        Padding(
            f"- {len(vsite_charge_parameter_keys)} charge increment and "
            f"{len(vsite_coordinate_parameter_keys)} coordinate parameters will be "
            f"trained",
            (0, 0, 0, 4),
        )
    )

    console.print("")
    console.rule("objective function")
    console.print("")

    # Compute the terms that will appear in the loss function and merge them together
    # for improved performance and convenience
    objective_terms_train = generate_objective_terms(
        esp_records_train,
        conformer_settings,
        charge_settings,
        bcc_collection,
        bcc_parameter_keys,
        vsite_collection,
        vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys,
        input_parameter_directory,
        n_processes,
    )

    console.print(
        f"the objective function is the sum of {len(objective_terms_train)} terms"
    )

    with console.status("merging objective terms"):
        objective_term_train = ESPObjectiveTerm.combine(*objective_terms_train)
        objective_term_train.to_backend("torch")

    console.print("")
    console.rule("training")
    console.print("")

    # Vectorize our BCC and virtual site parameters into flat tensors that can be
    # provided to and trained by a PyTorch optimizer.
    (
        current_charge_increments,
        current_vsite_coordinates,
        parameters_to_optimize,
    ) = vectorize_collections(
        bcc_collection,
        bcc_parameter_keys,
        vsite_collection,
        vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys,
    )

    # Optimize the parameters.
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=learning_rate)
    losses = []

    for epoch in track(list(range(n_epochs + 1)), description="training"):

        loss = objective_term_train.loss(
            current_charge_increments, current_vsite_coordinates
        )
        losses.append(float(loss.detach().numpy()))

        if current_vsite_coordinates is not None:

            loss += compute_vsite_pentalty(
                current_vsite_coordinates, vsite_coordinate_restraint_settings
            )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 5 == 0:
            console.print(f"epoch {epoch}: loss={loss.item()}")

    os.makedirs(output_directory, exist_ok=True)

    numpy.savetxt(
        os.path.join(output_directory, "train-losses.txt"), numpy.array(losses)
    )

    pyplot.plot(losses)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(output_directory, "train-losses.png"))

    final_bcc_collection, final_vsite_collection = copy_final_values(
        current_charge_increments,
        current_vsite_coordinates,
        bcc_collection,
        bcc_parameter_keys,
        vsite_collection,
        vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys,
    )

    with open(
        os.path.join(output_directory, "final-parameters-base.json"), "w"
    ) as file:
        json.dump((conformer_settings.dict(), charge_settings.dict()), file, indent=2)

    with open(os.path.join(output_directory, "final-parameters-bcc.json"), "w") as file:
        file.write(final_bcc_collection.json(indent=2))

    with open(
        os.path.join(output_directory, "final-parameters-v-site.json"), "w"
    ) as file:
        file.write(final_vsite_collection.json(indent=2))


if __name__ == "__main__":
    main()
