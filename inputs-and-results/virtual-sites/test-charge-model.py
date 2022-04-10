import functools
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy
import rich
from click_option_group import optgroup
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
    LibraryChargeParameter,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.charges.vsite import VirtualSiteCollection, VirtualSiteGenerator
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.utilities.geometry import compute_inverse_distance_matrix
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule
from openff.units import unit
from pydantic import parse_file_as
from rich.progress import track

_CACHED_CHARGES = {}


def compute_base_charge(
    molecule: Molecule,
    conformer_settings: ConformerSettings,
    charge_settings: QCChargeSettings,
):

    tagged_smiles = molecule.to_smiles(mapped=True)

    if tagged_smiles in _CACHED_CHARGES:
        return _CACHED_CHARGES[tagged_smiles]

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


def compute_test_molecule_rmse(
    esp_records: List[MoleculeESPRecord],
    charge_collection: Union[
        Tuple[ConformerSettings, QCChargeSettings], LibraryChargeCollection
    ],
    bcc_collection: BCCCollection,
    vsite_collection: VirtualSiteCollection,
) -> float:

    from simtk import unit as simtk_unit

    with capture_toolkit_warnings():

        molecule: Molecule = Molecule.from_mapped_smiles(
            esp_records[0].tagged_smiles, allow_undefined_stereo=True
        )

        if not isinstance(charge_collection, LibraryChargeCollection):

            conformer_settings, charge_settings = charge_collection
            charge_collection = compute_base_charge(
                molecule, conformer_settings, charge_settings
            )

        atom_charges = LibraryChargeGenerator.generate(
            molecule, charge_collection
        ) + BCCGenerator.generate(molecule, bcc_collection)
        vsite_charges = VirtualSiteGenerator.generate_charge_increments(
            molecule, vsite_collection
        )

        n_vsites = len(vsite_charges) - molecule.n_atoms

        full_charges = (
            numpy.vstack([atom_charges, numpy.zeros((n_vsites, 1))]) + vsite_charges
        )

        per_record_rmse = []

        for esp_record in esp_records:

            esp_molecule: Molecule = Molecule.from_mapped_smiles(
                esp_record.tagged_smiles, allow_undefined_stereo=True
            )
            esp_molecule._conformers = [esp_record.conformer * simtk_unit.angstrom]

            _, mapping = Molecule.are_isomorphic(
                esp_molecule, molecule, return_atom_map=True
            )
            esp_molecule.remap(mapping, current_to_new=True)

            [esp_conformer] = extract_conformers(esp_molecule)

            if len(vsite_collection.parameters) > 0:
                vsite_coordinates = VirtualSiteGenerator.generate_positions(
                    molecule, vsite_collection, esp_conformer
                )
                full_coordinates = numpy.vstack([esp_conformer, vsite_coordinates])
            else:
                full_coordinates = esp_conformer

            inverse_distance_matrix = compute_inverse_distance_matrix(
                esp_record.grid_coordinates, full_coordinates.m_as(unit.angstrom)
            )
            inverse_distance_matrix = unit.convert(
                inverse_distance_matrix, unit.angstrom ** -1, unit.bohr ** -1
            )

            delta = inverse_distance_matrix @ full_charges - esp_record.esp
            rmse = numpy.sqrt(numpy.mean(delta * delta))

            per_record_rmse.append(rmse)

    return float(numpy.mean(per_record_rmse))


def compute_test_rmse(
    esp_records: List[MoleculeESPRecord],
    charge_collection: Union[
        Tuple[ConformerSettings, QCChargeSettings], LibraryChargeCollection
    ],
    bcc_collection: BCCCollection,
    vsite_collection: Optional[VirtualSiteCollection],
    n_processes: int,
) -> Dict[str, float]:

    esp_records_by_smiles = defaultdict(list)

    with capture_toolkit_warnings():

        for esp_record in track(esp_records, "grouping records by molecule"):

            smiles = Molecule.from_smiles(
                esp_record.tagged_smiles, allow_undefined_stereo=True
            ).to_smiles(mapped=False)

            esp_records_by_smiles[smiles].append(esp_record)

    with Pool(processes=n_processes) as pool:

        per_molecule_rmse_func = functools.partial(
            compute_test_molecule_rmse,
            charge_collection=charge_collection,
            bcc_collection=bcc_collection,
            vsite_collection=vsite_collection,
        )

        smiles_list, smiles_esp_records = zip(*esp_records_by_smiles.items())

        rmse_list = list(
            track(
                pool.imap(per_molecule_rmse_func, smiles_esp_records),
                "computing per molecule RMSE",
                total=len(smiles_esp_records),
            )
        )

        per_molecule_rmse = {
            smiles: rmse for smiles, rmse in zip(smiles_list, rmse_list)
        }

    return per_molecule_rmse


@click.command()
@click.option(
    "--input-esp-records",
    "input_path",
    help="The file path to the input set of ESP records to test against.",
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
    "--output",
    "output_path",
    help="A path to the JSON file to save the average per molecule RMSE to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
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
    input_path,
    charge_collection_path,
    bcc_collection_path,
    vsite_collection_path,
    output_path,
    n_processes,
):

    console = rich.get_console()

    console.print("")
    console.rule("test data")
    console.print("")

    with open(input_path, "rb") as file:
        with console.status("loading test data"):
            esp_records_test = pickle.load(file)
        console.print(f"loaded {len(esp_records_test)} testing records")

    console.print("")
    console.rule("model parameters")
    console.print("")

    charge_collection = parse_file_as(
        Union[Tuple[ConformerSettings, QCChargeSettings], LibraryChargeCollection],
        charge_collection_path,
    )

    if bcc_collection_path is not None:
        bcc_collection = BCCCollection.parse_file(bcc_collection_path)
    else:
        bcc_collection = BCCCollection(parameters=[])

    if vsite_collection_path is not None:
        vsite_collection = VirtualSiteCollection.parse_file(vsite_collection_path)
    else:
        vsite_collection = VirtualSiteCollection(parameters=[])

    # Determine which parameters will be trained
    console.print("testing a charge model containing ...")

    console.print("")
    console.rule("testing")
    console.print("")

    per_molecule_rmse = compute_test_rmse(
        esp_records_test,
        charge_collection,
        bcc_collection,
        vsite_collection,
        n_processes,
    )

    with open(output_path, "w") as file:
        json.dump(per_molecule_rmse, file)


if __name__ == "__main__":
    main()
