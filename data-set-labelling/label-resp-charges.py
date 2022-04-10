import functools
import pathlib
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Tuple

import click
import numpy
import rich
from nagl.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
)
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit.topology import Molecule
from rich.progress import track


@functools.lru_cache
def molecule_from_mapped_smiles(smiles: str) -> Molecule:
    return Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)


def _compute_resp_charges(
    esp_record_and_molecules: List[Tuple[MoleculeESPRecord, Molecule]]
) -> MoleculeRecord:

    with capture_toolkit_warnings():

        esp_records, molecules = zip(*esp_record_and_molecules)

        molecule = molecules[0]
        charge_parameter = None
        try:
            charge_parameter = generate_resp_charge_parameter(
                esp_records, IterativeSolver()
            )

            charges = LibraryChargeGenerator.generate(
                molecule, LibraryChargeCollection(parameters=[charge_parameter])
            )
        except ChargeAssignmentError:

            with open("failed.pkl", "wb") as file:
                pickle.dump(esp_record_and_molecules, file)

            print("FAILED", molecule.to_smiles(), charge_parameter.smiles)

        return MoleculeRecord(
            smiles=molecule.to_smiles(mapped=True),
            conformers=[
                ConformerRecord(
                    coordinates=numpy.zeros((len(charges), 3)),
                    partial_charges=[
                        PartialChargeSet(
                            method="resp",
                            values=charges.flatten().tolist(),
                        )
                    ],
                )
            ],
        )


@click.option(
    "--input",
    "input_path",
    help="The path (.pkl) to the pickled ESP records.",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path (.sqlite) to the store to store the charges in.",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
)
@click.command()
def main(input_path, output_path):

    console = rich.get_console()

    with open(input_path, "rb") as file:
        all_esp_records = pickle.load(file)["esp-records"]

    esp_records_by_smiles = defaultdict(list)

    with capture_toolkit_warnings():

        for esp_record in track(
            all_esp_records, description="sorting records", transient=True
        ):

            molecule = molecule_from_mapped_smiles(esp_record.tagged_smiles)
            smiles = molecule.to_smiles(isomeric=True, mapped=False)

            esp_records_by_smiles[smiles].append((esp_record, molecule))

    with Pool(processes=10) as pool:

        charge_records = list(
            track(
                pool.imap(_compute_resp_charges, esp_records_by_smiles.values()),
                description="computing charges",
                total=len(esp_records_by_smiles),
            )
        )

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with console.status("storing RESP charges"):
        record_store = MoleculeStore(output_path)
        record_store.store(*charge_records)


if __name__ == "__main__":
    main()
