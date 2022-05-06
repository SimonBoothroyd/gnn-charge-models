import pickle
from pathlib import Path
from typing import Tuple, Union

import click
import numpy
import rich
from molesp.cli._cli import compute_surface
from molesp.models import ESPMolecule, Surface
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
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import MSKGridSettings
from openff.recharge.utilities.geometry import compute_inverse_distance_matrix
from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii
from openff.toolkit.topology import Molecule
from openff.units import unit
from openff.utilities import temporary_cd
from openmm import unit as openmm_unit
from pydantic import parse_file_as

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


def compute_mm_esp(
    molecule: Molecule,
    conformer: unit.Quantity,
    charge_collection: Union[
        Tuple[ConformerSettings, QCChargeSettings], LibraryChargeCollection
    ],
    bcc_collection: BCCCollection,
    vsite_collection: VirtualSiteCollection,
    grid: unit.Quantity,
):

    console = rich.get_console()
    console.print("applying MM charges")

    if not isinstance(charge_collection, LibraryChargeCollection):
        conformer_settings, charge_settings = charge_collection
        charge_collection = compute_base_charge(
            molecule, conformer_settings, charge_settings
        )

    atom_charges = LibraryChargeGenerator.generate(molecule, charge_collection)

    if len(bcc_collection.parameters) > 0:
        atom_charges += BCCGenerator.generate(molecule, bcc_collection)
    if len(vsite_collection.parameters) > 0:
        vsite_charges = VirtualSiteGenerator.generate_charge_increments(
            molecule, vsite_collection
        )
        n_vsites = len(vsite_charges) - molecule.n_atoms

        full_charges = (
            numpy.vstack([atom_charges, numpy.zeros((n_vsites, 1))]) + vsite_charges
        )
    else:
        full_charges = atom_charges

    if len(vsite_collection.parameters) > 0:
        vsite_coordinates = VirtualSiteGenerator.generate_positions(
            molecule, vsite_collection, conformer
        )
        full_coordinates = numpy.vstack([conformer, vsite_coordinates])
    else:
        full_coordinates = conformer

    console.print(f"computing MM ESP on grid with {len(grid)} points")

    inverse_distance_matrix = compute_inverse_distance_matrix(
        grid.m_as(unit.angstrom), full_coordinates.m_as(unit.angstrom)
    )
    inverse_distance_matrix = unit.convert(
        inverse_distance_matrix, unit.angstrom**-1, unit.bohr**-1
    )

    return (inverse_distance_matrix @ full_charges) * (
        unit.hartree / unit.e
    ), full_coordinates


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
    "output_directory",
    help="The directory to save the cube files in.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
)
def main(
    input_path,
    charge_collection_path,
    bcc_collection_path,
    vsite_collection_path,
    output_directory,
):

    console = rich.get_console()

    console.print("")
    console.rule("QC ESP data")
    console.print("")

    with open(input_path, "rb") as file:
        with console.status("loading data"):
            esp_records = pickle.load(file)
        console.print(f"loaded {len(esp_records)} records")

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
        with capture_toolkit_warnings():
            vsite_collection = VirtualSiteCollection.parse_file(vsite_collection_path)
    else:
        vsite_collection = VirtualSiteCollection(parameters=[])

    console.print("model contains ...")

    console.print("")
    console.rule("cube files")
    console.print("")

    with capture_toolkit_warnings():

        for i, esp_record in enumerate(esp_records):

            molecule: Molecule = Molecule.from_mapped_smiles(
                esp_record.tagged_smiles, allow_undefined_stereo=True
            )

            conformer = esp_record.conformer_quantity
            conformer -= numpy.mean(conformer, axis=0)

            molecule._conformers = [
                conformer.m_as(unit.angstrom) * openmm_unit.angstrom
            ]

            with console.status("generating surface"):

                vdw_radii = compute_vdw_radii(molecule, radii_type=VdWRadiiType.Bondi)
                radii = (
                    numpy.array([[radii] for radii in vdw_radii.m_as(unit.angstrom)])
                    * unit.angstrom
                )

                vertices, indices = compute_surface(
                    molecule, conformer, radii, 1.4, 0.2 * unit.angstrom
                )

            with console.status("generating MM ESP"):
                mm_esp, mm_coordinates = compute_mm_esp(
                    molecule,
                    conformer,
                    charge_collection,
                    bcc_collection,
                    vsite_collection,
                    vertices * unit.angstrom,
                )
                n_v_sites = len(mm_coordinates) - len(conformer)

            with console.status("generating QC ESP"):

                esp_settings = ESPSettings(
                    basis="6-31G*", method="hf", grid_settings=MSKGridSettings()
                )

                with temporary_cd():
                    _, qc_esp, _ = Psi4ESPGenerator._generate(
                        molecule,
                        conformer,
                        vertices * unit.angstrom,
                        esp_settings,
                        "",
                        minimize=False,
                        compute_esp=True,
                        compute_field=False,
                    )

            molecule_output_directory = Path(output_directory, str(i))
            molecule_output_directory.mkdir(parents=True, exist_ok=True)

            esp_molecule = ESPMolecule(
                atomic_numbers=[atom.atomic_number for atom in molecule.atoms]
                + ([0] * n_v_sites),
                conformer=mm_coordinates.m_as(unit.angstrom).flatten().tolist(),
                surface=Surface(
                    vertices=vertices.flatten().tolist(),
                    indices=indices.flatten().tolist(),
                ),
                esp={
                    "QC ESP": qc_esp.m_as(unit.hartree / unit.e).flatten().tolist(),
                    "MM ESP": mm_esp.m_as(unit.hartree / unit.e).flatten().tolist(),
                    "QC - MM": (qc_esp - mm_esp)
                    .m_as(unit.hartree / unit.e)
                    .flatten()
                    .tolist(),
                },
            )

            with Path(molecule_output_directory, "mol.pkl").open("wb") as file:
                pickle.dump(esp_molecule, file)


if __name__ == "__main__":
    main()
