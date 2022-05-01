import functools
import pickle
from typing import List

import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
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
            ]
        ]

        test_smiles = set.intersection(*smiles_per_charge_model)

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
