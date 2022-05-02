"""Train a set of BCC + v-site parameters"""
import copy
import functools
import hashlib
import json
import os.path
import pickle
import sqlite3
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

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


class ObjectiveTermCache:
    def __init__(
        self, file_path: Union[os.PathLike, str], clear_existing: bool = False
    ):

        self._file_path = file_path

        self._connection = sqlite3.connect(
            file_path, detect_types=sqlite3.PARSE_DECLTYPES
        )

        sqlite3.register_adapter(ESPObjectiveTerm, pickle.dumps)
        sqlite3.register_converter("pickle", pickle.loads)

        self._create_schema()

        if clear_existing:
            self.clear()

    def __del__(self):

        if self._connection is not None:
            self._connection.close()

    def _create_schema(self):

        with self._connection:

            self._connection.execute(
                "create table if not exists info (version integer)"
            )
            db_info = self._connection.execute("select * from info").fetchall()

            if len(db_info) == 0:
                self._connection.execute("insert into info values(1)")
            else:
                assert len(db_info) == 1 and db_info[0] == (1,)

            self._connection.execute(
                "create table if not exists cache "
                "(id integer primary key, hash text, term pickle)"
            )
            self._connection.execute(
                "create index if not exists ix_hash on cache(hash)"
            )

            self._connection.execute("pragma optimize")

    def create(self, terms: Iterable[Tuple[str, ESPObjectiveTerm]]):

        with self._connection:

            self._connection.executemany(
                "insert into cache (hash, term) values (?, ?)", terms
            )
            self._connection.execute("pragma optimize")

    def read_all(
        self, skip: Optional[int] = None, limit: Optional[int] = None
    ) -> List[ESPObjectiveTerm]:

        statement = "select term from cache"

        bindings = []

        if limit is not None:
            statement += " limit ?"
            bindings.append(limit)
        if skip is not None:
            statement += " offset ?"
            bindings.append(skip)

        return [
            esp_record
            for (esp_record,) in self._connection.execute(
                statement, bindings
            ).fetchall()
        ]

    def read(self, term_hash: str) -> Optional[ESPObjectiveTerm]:

        return_value = self._connection.execute(
            "select term from cache where hash=?", (term_hash,)
        ).fetchone()

        if return_value is None:
            return None

        (esp_record,) = return_value
        return esp_record

    def clear(self):

        with self._connection:
            self._connection.execute("delete from molecules")


def hash_record(record: MoleculeESPRecord) -> str:
    def numpy_encoder(obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return pydantic_encoder(obj)

    return hashlib.sha256(
        record.json(encoder=numpy_encoder).encode("utf-8")
    ).hexdigest()


def compute_base_charge(
    tagged_smiles,
    conformer_settings: ConformerSettings,
    charge_settings: QCChargeSettings,
) -> LibraryChargeParameter:

    if tagged_smiles not in _CACHED_CHARGES:

        molecule = Molecule.from_mapped_smiles(
            tagged_smiles, allow_undefined_stereo=True
        )

        conformers = ConformerGenerator.generate(molecule, conformer_settings)
        charges = QCChargeGenerator.generate(molecule, conformers, charge_settings)

        _CACHED_CHARGES[tagged_smiles] = LibraryChargeParameter(
            smiles=tagged_smiles,
            value=[float(v) for v in charges.flatten().tolist()],
        )

    return _CACHED_CHARGES[tagged_smiles]


def library_charge_to_smiles(
    parameter: LibraryChargeParameter,
) -> Tuple[str, LibraryChargeParameter]:

    with capture_toolkit_warnings():
        return (
            Molecule.from_smiles(
                parameter.smiles, allow_undefined_stereo=True
            ).to_smiles(mapped=False),
            parameter,
        )


def library_charges_to_dict(
    charge_collection: LibraryChargeCollection, pool: Pool
) -> Dict[str, LibraryChargeParameter]:

    return_value = dict(
        track(
            pool.imap(library_charge_to_smiles, charge_collection.parameters),
            total=len(charge_collection.parameters),
            description="processing library charge collection",
        )
    )
    assert len(return_value) == len(charge_collection.parameters)
    return return_value


def generate_objective_term(
    esp_record: MoleculeESPRecord,
    charge_collection: Union[
        Dict[str, LibraryChargeParameter], Tuple[ConformerSettings, QCChargeSettings]
    ],
    bcc_collection: BCCCollection,
    bcc_parameter_keys: List[str],
    vsite_collection: VirtualSiteCollection,
    vsite_charge_parameter_keys: List[VirtualSiteChargeKey],
    vsite_coordinate_parameter_keys: List[VirtualSiteGeometryKey],
) -> Tuple[str, ESPObjectiveTerm]:

    with capture_toolkit_warnings():

        if isinstance(charge_collection, dict):
            record_smiles = Molecule.from_smiles(
                esp_record.tagged_smiles, allow_undefined_stereo=True
            ).to_smiles(mapped=False)
            charge_parameter = charge_collection[record_smiles]
        else:
            charge_parameter = compute_base_charge(
                esp_record.tagged_smiles, charge_collection[0], charge_collection[1]
            )

        # Matching a full charge collection is waaaaaay toooooo sllllooooooowwwwwwww
        # so we create a subset for faster matching. Upstream fixes needed....
        charge_collection_subset = LibraryChargeCollection(
            parameters=[charge_parameter]
        )

        objective_term_generator = ESPObjective.compute_objective_terms(
            esp_records=[esp_record],
            charge_collection=charge_collection_subset,
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

        return (
            hash_record(esp_record),
            cast(ESPObjectiveTerm, list(objective_term_generator)[0]),
        )


def generate_objective_terms(
    esp_records: List[MoleculeESPRecord],
    charge_collection: Union[
        LibraryChargeCollection, Tuple[ConformerSettings, QCChargeSettings]
    ],
    bcc_collection: BCCCollection,
    bcc_parameter_keys: List[str],
    vsite_collection: VirtualSiteCollection,
    vsite_charge_parameter_keys: List[VirtualSiteChargeKey],
    vsite_coordinate_parameter_keys: List[VirtualSiteGeometryKey],
    cache_directory: str,
    n_processes: int,
) -> ObjectiveTermCache:

    console = rich.get_console()

    with Pool(processes=n_processes) as pool:

        with console.status("creating objective cache hash"):

            cache_hash_input = json.dumps(
                [
                    charge_collection.json()
                    if isinstance(charge_collection, LibraryChargeCollection)
                    else (charge_collection[0].json(), charge_collection[1].json()),
                    bcc_collection.json(),
                    bcc_parameter_keys,
                    vsite_collection.json(),
                    vsite_charge_parameter_keys,
                    vsite_coordinate_parameter_keys,
                ],
                sort_keys=True,
            )

        cache_hash = hashlib.sha256(cache_hash_input.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_directory, f"train-objective-{cache_hash}.sqlite"
        )

        if not os.path.isfile(cache_path):
            console.print(
                f"creating cache at [repr.filename]{cache_path}[/repr.filename]"
            )

        cache = ObjectiveTermCache(cache_path, clear_existing=False)
        uncached_records = []

        esp_record_hashes = list(
            track(
                pool.imap(hash_record, esp_records),
                "hashing ESP records",
                total=len(esp_records),
            )
        )

        for esp_record_hash, esp_record in track(
            zip(esp_record_hashes, esp_records),
            "reading records from cache",
            total=len(esp_records),
        ):

            cached_term = cache.read(esp_record_hash)

            if cached_term is None:
                uncached_records.append(esp_record)
                continue

        processed_charge_collection = (
            library_charges_to_dict(charge_collection, pool)
            if isinstance(charge_collection, LibraryChargeCollection)
            else charge_collection
        )

        for i, (term_hash, term) in enumerate(
            track(
                pool.imap(
                    functools.partial(
                        generate_objective_term,
                        charge_collection=processed_charge_collection,
                        bcc_collection=bcc_collection,
                        bcc_parameter_keys=bcc_parameter_keys,
                        vsite_collection=vsite_collection,
                        vsite_charge_parameter_keys=vsite_charge_parameter_keys,
                        vsite_coordinate_parameter_keys=vsite_coordinate_parameter_keys,
                    ),
                    uncached_records,
                ),
                total=len(uncached_records),
                description="building objective terms",
                transient=True,
            )
        ):

            if i % 1000 == 0:
                console.print(f"generated {i} / {len(uncached_records)}")

            cache.create([(term_hash, term)])

    return cache


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

    with capture_toolkit_warnings():

        with console.status("loading model parameters"):

            charge_collection = parse_file_as(
                Union[
                    LibraryChargeCollection, Tuple[ConformerSettings, QCChargeSettings]
                ],
                os.path.join(input_parameter_directory, "initial-parameters-base.json"),
            )
            bcc_collection = BCCCollection.parse_file(
                os.path.join(input_parameter_directory, "initial-parameters-bcc.json")
            )
            vsite_collection = VirtualSiteCollection.parse_file(
                os.path.join(
                    input_parameter_directory, "initial-parameters-v-site.json"
                )
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
        and (
            parameter.provenance is None
            or parameter.provenance["code"][:2] != parameter.provenance["code"][-2:]
        )
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
    objective_term_cache = generate_objective_terms(
        esp_records_train,
        charge_collection,
        bcc_collection,
        bcc_parameter_keys,
        vsite_collection,
        vsite_charge_parameter_keys,
        vsite_coordinate_parameter_keys,
        input_parameter_directory,
        n_processes,
    )
    objective_terms = objective_term_cache.read_all()

    console.print(f"the objective function is the sum of {len(objective_terms)} terms")

    with console.status("merging objective terms"):
        objective_term = ESPObjectiveTerm.combine(*objective_terms)

        if objective_term.vsite_coord_assignment_matrix is not None:
            objective_term.vsite_coord_assignment_matrix = (
                objective_term.vsite_coord_assignment_matrix.astype(int, copy=False)
            )

        objective_term.to_backend("torch")

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

        loss = objective_term.loss(current_charge_increments, current_vsite_coordinates)
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
        json.dump(
            charge_collection.dict()
            if isinstance(charge_collection, LibraryChargeCollection)
            else (
                charge_collection[0].dict(),
                charge_collection[1].dict(),
            ),
            file,
            indent=2,
        )

    with open(os.path.join(output_directory, "final-parameters-bcc.json"), "w") as file:
        file.write(final_bcc_collection.json(indent=2))

    with open(
        os.path.join(output_directory, "final-parameters-v-site.json"), "w"
    ) as file:
        file.write(final_vsite_collection.json(indent=2))


if __name__ == "__main__":
    main()
