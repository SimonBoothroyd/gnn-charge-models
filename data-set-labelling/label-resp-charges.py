import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import click
import numpy
import psiresp
from openff.recharge.esp.qcresults import from_qcportal_results
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.units import unit
from psiresp import Conformer, Orientation
from qcportal.models import KeywordSet
from qcportal.models import Molecule as QCMolecule
from qcportal.models import ResultRecord as QCRecord
from tqdm import tqdm


def _compute_resp_charges(
    input_tuple: Tuple[str, List[Tuple[QCRecord, QCMolecule]]],
    qc_keywords: Dict[str, KeywordSet],
) -> Tuple[str, numpy.ndarray]:

    cmiles, qc_results = input_tuple

    _, qc_molecule = qc_results[0]
    resp_molecule = psiresp.Molecule(qcmol=qc_molecule)

    for qc_record, qc_molecule in qc_results:

        qc_keyword_set = qc_keywords[qc_record.keywords]

        esp_record: MoleculeESPRecord = from_qcportal_results(
            qc_record,
            qc_molecule,
            qc_keyword_set,
            MSKGridSettings(),
            compute_field=False,
        )

        orientation = Orientation(qcmol=qc_molecule)
        orientation.grid = esp_record.grid_coordinates_quantity.to(unit.angstrom).m
        orientation.esp = esp_record.esp_quantity.to(unit.hartree / unit.e).m.flatten()

        conformer = Conformer(qcmol=qc_molecule)
        conformer.orientations.append(orientation)

        resp_molecule.conformers.append(conformer)

    job = psiresp.Job(molecules=[resp_molecule])
    charges = job.compute_charges()[0]

    return cmiles, charges


@click.option(
    "--input",
    "input_path",
    help="The path (.pkl) to the saved QC data to derive the RESP charges from.",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "--output",
    "output_directory",
    help="The directory to save the RESP charges in.",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
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
def main(input_path, output_directory, batch_size, batch_index):

    with open(input_path, "rb") as file:
        qc_results, qc_keywords = pickle.load(file)

    qc_results_per_molecule = defaultdict(list)

    for qc_record, qc_molecule in tqdm(qc_results):

        cmiles = qc_molecule.extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        qc_results_per_molecule[cmiles].append((qc_record, qc_molecule))

    batch_cmiles = sorted(qc_results_per_molecule)[
        batch_index * batch_size : (batch_index + 1) * batch_size
    ]

    batch_qc_results = {
        cmiles: qc_results_per_molecule[cmiles] for cmiles in batch_cmiles
    }

    charges = list(
        tqdm(
            (
                _compute_resp_charges(input_tuple, qc_keywords=qc_keywords)
                for input_tuple in batch_qc_results.items()
            ),
            desc="RESP charges",
            total=len(batch_qc_results),
        )
    )

    os.makedirs(output_directory, exist_ok=True)

    output_name = (
        f"{os.path.splitext(os.path.basename(input_path))[0]}-{batch_index}.pkl"
    )

    with open(os.path.join(output_directory, output_name), "wb") as file:
        pickle.dump(charges, file)


if __name__ == "__main__":
    main()
