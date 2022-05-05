import functools
import pickle
from multiprocessing import Pool
from typing import List, Tuple

import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.library import LibraryChargeCollection
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit.topology import Molecule
from rich import pretty
from rich.progress import track


@functools.lru_cache()
def strip_map_indices(smiles: str) -> str:
    return Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_smiles(
        mapped=False
    )


def has_valid_bcc(smiles: str, bcc_collection: BCCCollection) -> Tuple[str, bool]:

    with capture_toolkit_warnings():

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        try:
            BCCGenerator.applied_corrections(molecule, bcc_collection=bcc_collection)
        except ChargeAssignmentError:
            return smiles, False

    return smiles, True


def main():

    console = rich.get_console()
    pretty.install(console)

    with capture_toolkit_warnings():

        smiles_per_charge_model = [
            {
                strip_map_indices(parameter.smiles)
                for parameter in track(
                    LibraryChargeCollection.parse_file(charge_model_path).parameters,
                    description=f"loading {charge_model_path} SMILES",
                )
            }
            for charge_model_path in [
                "reference-charges/am1-charge-industry-set.json",
                "reference-charges/am1bcc-charge-industry-set.json",
                "reference-charges/resp-charges-industry-set.json",
                "gnn-charge-models/gnn-am1-charges-base.json",
            ]
        ]

        filtered_smiles = set.intersection(*smiles_per_charge_model)

        bcc_collection = BCCCollection.parse_file(
            "../train-charge-models/gnn-charge-models/gnn-am1-v2-bcc/lr-0.0025-n-400/final-parameters-bcc.json"
        )

        test_smiles = []

        with Pool(processes=10) as pool:

            for smiles, is_valid in track(
                pool.imap(
                    functools.partial(has_valid_bcc, bcc_collection=bcc_collection),
                    filtered_smiles,
                ),
                description="filtering by BCC",
                total=len(filtered_smiles),
            ):

                if not is_valid:
                    continue

                test_smiles.append(smiles)

        with open(
            "../data-set-labelling/qc-esp/esp-records-industry-set.pkl", "rb"
        ) as file:
            esp_records: List[MoleculeESPRecord] = pickle.load(file)["esp-records"]

        test_esp_records = [
            esp_record
            for esp_record in track(esp_records, "filtering records")
            if strip_map_indices(esp_record.tagged_smiles) in test_smiles
        ]

    with open("test-esp-records.pkl", "wb") as file:
        pickle.dump(test_esp_records, file)


if __name__ == "__main__":
    main()
