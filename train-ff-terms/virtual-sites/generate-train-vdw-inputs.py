import copy
import os
from typing import Tuple, Union

import click
import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from nonbonded.library.factories.inputs.optimization import OptimizationInputFactory
from nonbonded.library.models.datasets import DataSet
from nonbonded.library.models.engines import ForceBalance
from nonbonded.library.models.forcefield import ForceField, Parameter
from nonbonded.library.models.projects import Optimization
from nonbonded.library.models.targets import EvaluatorTarget
from openff.recharge.charges.bcc import BCCCollection
from openff.recharge.charges.library import LibraryChargeCollection
from openff.recharge.charges.qc import QCChargeSettings
from openff.recharge.charges.vsite import VirtualSiteCollection
from openff.recharge.conformers import ConformerSettings
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines import smirnoff
from openff.utilities import temporary_cd
from pydantic import parse_file_as
from rich import pretty
from rich.console import NewLine
from rich.progress import track

ChargeCollection = Union[
    LibraryChargeCollection, Tuple[ConformerSettings, QCChargeSettings]
]


def load_charge_model(
    charge_collection_path: str, bcc_collection_path: str, vsite_collection_path: str
) -> Tuple[ChargeCollection, BCCCollection, VirtualSiteCollection]:

    charge_collection = parse_file_as(ChargeCollection, charge_collection_path)

    if bcc_collection_path is not None:
        bcc_collection = BCCCollection.parse_file(bcc_collection_path)
    else:
        bcc_collection = BCCCollection(parameters=[])

    if vsite_collection_path is not None:
        vsite_collection = VirtualSiteCollection.parse_file(vsite_collection_path)
    else:
        vsite_collection = VirtualSiteCollection(parameters=[])

    return charge_collection, bcc_collection, vsite_collection


def add_charge_model_to_force_field(
    force_field: smirnoff.ForceField,
    training_set: DataSet,
    charge_collection: ChargeCollection,
    bcc_collection: BCCCollection,
    vsite_collection: VirtualSiteCollection,
) -> smirnoff.ForceField:

    console = rich.get_console()
    console.print(NewLine())

    force_field = copy.deepcopy(force_field)

    unique_smiles = {
        Molecule.from_smiles(component.smiles, allow_undefined_stereo=True).to_smiles(
            mapped=True
        )
        for entry in training_set.entries
        for component in entry.components
    }

    console.print("- adding charge model to force field")
    console.print(f"- mapping {len(unique_smiles)} unique molecules to library charges")

    # library_charges: List[LibraryChargeParameter] = []

    water: Molecule = Molecule.from_smiles("O")

    for smiles in track(unique_smiles, ""):

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        is_water, _ = Molecule.are_isomorphic(molecule, water)

        if is_water:
            console.print(f"skipping {smiles}")
            continue

        if "atom_map" in molecule.properties:
            del molecule.properties["atom_map"]

        if isinstance(charge_collection, LibraryChargeCollection):
            raise NotImplementedError()
        # Library charges cause OOM issues on Lilac when applied using the TK...
        #     charges = LibraryChargeGenerator.generate(molecule, charge_collection)
        # else:
        #     conformer_settings, charge_settings = charge_collection
        #
        #     conformers = ConformerGenerator.generate(molecule, conformer_settings)
        #     charges = QCChargeGenerator.generate(molecule, conformers, charge_settings)
        #
        # charges += BCCGenerator.generate(molecule, bcc_collection)
        #
        # library_charges.append(
        #     LibraryChargeParameter(
        #         smiles=molecule.to_smiles(mapped=True),
        #         value=[float(v) for v in cast(List[float], charges.flatten().tolist())],
        #     )
        # )

    # atom_charge_collection = LibraryChargeCollection(parameters=library_charges)

    force_field.deregister_parameter_handler("ToolkitAM1BCC")
    force_field.register_parameter_handler(bcc_collection.to_smirnoff())
    force_field.register_parameter_handler(vsite_collection.to_smirnoff())

    # library_charge_handler: LibraryChargeHandler = force_field.get_parameter_handler(
    #     "LibraryCharges"
    # )
    #
    # for parameter in atom_charge_collection.to_smirnoff().parameters:
    #     library_charge_handler.add_parameter(parameter=parameter)

    return force_field


@click.command()
@click.option(
    "--input-training-set",
    "training_set_path",
    help="The path to the training set",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-parameters-base",
    "charge_collection_path",
    help="The path to the base charge model parameters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-parameters-bcc",
    "bcc_collection_path",
    help="The path to the BCC parameters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    "--input-parameters-v-site",
    "vsite_collection_path",
    help="The path to the BCC parameters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    "--output-name",
    "output_name",
    help="The name of the optimize to save.",
    type=str,
    required=True,
)
@click.option(
    "--output-dir",
    "output_directory",
    help="The directory to save the inputs in.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="",
    show_default=True,
)
def main(
    training_set_path,
    charge_collection_path,
    bcc_collection_path,
    vsite_collection_path,
    output_name,
    output_directory,
):

    console = rich.get_console()
    pretty.install(console)

    training_set = DataSet.parse_file(training_set_path)

    with capture_toolkit_warnings():

        with console.status("loading charge model"):
            charge_collection, bcc_collection, vsite_collection = load_charge_model(
                charge_collection_path, bcc_collection_path, vsite_collection_path
            )
        console.print("- loaded charge model")

        base_force_field = smirnoff.ForceField("openff-2.0.0.offxml")

        force_field = add_charge_model_to_force_field(
            base_force_field,
            training_set,
            charge_collection,
            bcc_collection,
            vsite_collection,
        )

    optimization = Optimization(
        project_id="ng-charge-models",
        study_id="virtual-sites",
        id=output_name,
        name=output_name,
        description=(
            "An optimization of the vdW parameters of the `openff-1.3.0` force "
            "field against a training set of physical property data."
        ),
        engine=ForceBalance(
            priors={
                "vdW/Atom/epsilon": 0.1,
                "vdW/Atom/rmin_half": 1.0,
            }
        ),
        targets=[
            EvaluatorTarget(
                id="phys-prop",
                denominators={
                    "Density": "0.05 g / ml",
                    "EnthalpyOfMixing": "1.6 kJ / mol",
                },
                data_set_ids=["sage-train-v1"],
            )
        ],
        force_field=ForceField.from_openff(force_field),
        parameters_to_train=[
            Parameter(handler_type="vdW", attribute_name=attribute, smirks=smirks)
            for attribute in ["epsilon", "rmin_half"]
            for smirks in [
                "[#16:1]",
                "[#17:1]",
                "[#1:1]-[#6X3]",
                "[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]",
                "[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]",
                "[#1:1]-[#6X4]",
                "[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]",
                "[#1:1]-[#7]",
                "[#1:1]-[#8]",
                "[#35:1]",
                "[#6:1]",
                "[#6X4:1]",
                "[#7:1]",
                "[#8:1]",
                "[#8X2H0+0:1]",
                "[#8X2H1+0:1]",
            ]
        ],
        analysis_environments=[],
        max_iterations=5,
    )

    os.makedirs(output_directory, exist_ok=True)

    with temporary_cd(output_directory):
        OptimizationInputFactory.generate(
            optimization,
            conda_environment="gnn-charge-models",
            max_time="168:00",
            evaluator_preset="lilac-dask",
            evaluator_port=8000,
            n_evaluator_workers=60,
            reference_data_sets=[training_set],
        )


if __name__ == "__main__":
    main()
