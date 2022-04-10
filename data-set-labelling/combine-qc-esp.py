import os.path
import pickle
from glob import glob
from typing import List, Optional

import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.esp.storage import MoleculeESPRecord
from rich.padding import Padding


def load_esp_record(file_path: str) -> Optional[List[MoleculeESPRecord]]:

    if not os.path.isfile(file_path):
        return None

    with open(file_path, "rb") as file:
        return pickle.load(file)


def combine_qc_esp_records(console: "rich.Console", set_name: str):

    with open(f"qc-esp/{set_name}-set-conda-env.yml") as file:
        conda_env = file.read()

    with console.status("loading original QC records"):

        with open(f"qc-esp/qc-records-{set_name}.pkl", "rb") as file:
            qc_records, _ = pickle.load(file)

    n_qc_esp_batch_files = len(glob(f"qc-esp/{set_name}-set/msk-default-*.pkl"))

    with console.status("loading QC ESP records"):

        qc_esp_batch_files = [
            f"qc-esp/{set_name}-set/msk-default-{i + 1}.pkl"
            for i in range(n_qc_esp_batch_files)
        ]
        qc_esp_records_per_batch = {
            i: load_esp_record(qc_esp_file)
            for i, qc_esp_file in enumerate(qc_esp_batch_files)
        }
        qc_esp_records_per_batch = {
            i: records
            for i, records in qc_esp_records_per_batch.items()
            if records is not None
        }

    missing_batches = set(range(n_qc_esp_batch_files)) - {*qc_esp_records_per_batch}
    print(f"{len(missing_batches)} MISSING BATCHES: ", missing_batches)

    qc_esp_records = [
        qc_esp_record
        for batch in range(n_qc_esp_batch_files)
        for qc_esp_record in qc_esp_records_per_batch[batch]
    ]

    print(f"N QC RECORDS {len(qc_records)}  N QC ESP RECORDS {len(qc_esp_records)}")
    assert len(qc_records) == len(qc_esp_records)

    with console.status("storing QC ESP records"):

        esp_store_path = os.path.join("qc-esp", f"esp-records-{set_name}-set.pkl")

        with open(esp_store_path, "wb") as file:
            pickle.dump(
                {
                    "esp-records": qc_esp_records,
                    "provenance": {"conda-env": conda_env, "grid-type": "msk-default"},
                },
                file,
            )


def main():

    console = rich.get_console()

    with capture_toolkit_warnings():

        for set_name in ["fragment", "industry"]:

            console.print(Padding(f"processing {set_name} set", (1, 0, 1, 0)))
            combine_qc_esp_records(console, set_name)
            console.print(Padding(f"processed {set_name} set", (1, 0, 0, 0)))


if __name__ == "__main__":
    main()
