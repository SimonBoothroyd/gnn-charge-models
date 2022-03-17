import pickle
from typing import Dict, Tuple

import click
import psi4
from openff.recharge.esp.qcresults import from_qcportal_results
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import GridSettingsType
from pydantic import parse_file_as
from qcportal.models import KeywordSet
from qcportal.models import Molecule as QCMolecule
from qcportal.models import ResultRecord as QCRecord
from tqdm import tqdm


def _compute_esp(
    qc_result: Tuple[QCRecord, QCMolecule],
    qc_keywords: Dict[str, KeywordSet],
    grid_settings: GridSettingsType,
    n_threads: int,
    memory: str,
    compute_field: bool = True,
) -> MoleculeESPRecord:

    qc_record, qc_molecule = qc_result
    qc_keyword_set = qc_keywords[qc_record.keywords]

    psi4.core.be_quiet()
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)

    esp_record: MoleculeESPRecord = from_qcportal_results(
        qc_record,
        qc_molecule,
        qc_keyword_set,
        grid_settings,
        compute_field=compute_field,
    )

    return esp_record


@click.option(
    "--input",
    "input_path",
    help="The path (.pkl) to the saved QC data to reconstruct the QC ESP from.",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "--grid",
    "grid_path",
    help="The path to the JSON serialized grid settings to use.",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "--n-threads",
    help="The number of threads ( / CPUs ) to make available to psi4",
    type=int,
    required=True,
)
@click.option(
    "--memory",
    help="The amount of memory, provided as a string with units, to make available to "
    "psi4",
    type=str,
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path (.pkl) to save the ESP records to.",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    help="The size of the batch to compute.",
    show_default=True,
)
@click.option(
    "--batch-idx",
    "batch_index",
    type=int,
    default=0,
    help="The (zero-based) index of the batch to compute.",
    show_default=True,
)
@click.command()
def main(
    input_path, grid_path, n_threads, memory, output_path, batch_size, batch_index
):

    with open(input_path, "rb") as file:
        qc_results, qc_keywords = pickle.load(file)

    batch_qc_results = qc_results[
        batch_index * batch_size : (batch_index + 1) * batch_size
    ]

    grid_settings = parse_file_as(GridSettingsType, grid_path)

    esp_records = list(
        tqdm(
            (
                _compute_esp(
                    input_tuple,
                    qc_keywords=qc_keywords,
                    grid_settings=grid_settings,
                    n_threads=n_threads,
                    memory=memory,
                    compute_field=True,
                )
                for input_tuple in batch_qc_results
            ),
            desc="ESP",
            total=len(batch_qc_results),
        )
    )

    with open(output_path, "wb") as file:
        pickle.dump(esp_records, file)


if __name__ == "__main__":
    main()
