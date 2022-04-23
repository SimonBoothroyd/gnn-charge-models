import functools
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Tuple

import click
import numpy
import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.charges.vsite import VirtualSiteCollection, VirtualSiteGenerator
from openff.toolkit.topology import Molecule
from rich.progress import track


@functools.lru_cache()
def to_canonical_smiles(smiles: str) -> str:
    with capture_toolkit_warnings():
        return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_smiles(
            isomeric=True, mapped=False
        )


def process_molecule(
    smiles: str, vsite_collection: VirtualSiteCollection, bcc_collection: BCCCollection
) -> Tuple[str, bool, List[str], bool, List[str]]:

    with capture_toolkit_warnings():

        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        vsite_index_to_smirks = {}
        counter = 0

        for parameter in vsite_collection.parameters:
            for i in range(len(parameter.charge_increments)):
                vsite_index_to_smirks[counter] = parameter.smirks
                counter += 1

        vsite_assignment_matrix = VirtualSiteGenerator.build_charge_assignment_matrix(
            molecule, vsite_collection
        )
        applied_vsite_keys = [
            vsite_index_to_smirks[i]
            for i in range(vsite_assignment_matrix.shape[1])
            if numpy.any(vsite_assignment_matrix[:, i] != 0)
        ]

        try:
            applied_bcc_keys = [
                parameter.smirks
                for parameter in BCCGenerator.applied_corrections(
                    molecule, bcc_collection=bcc_collection
                )
            ]
            has_missing_bcc = False
        except ChargeAssignmentError:
            applied_bcc_keys = []
            has_missing_bcc = True

        return (
            smiles,
            len(applied_vsite_keys) > 0,
            applied_vsite_keys,
            has_missing_bcc,
            applied_bcc_keys,
        )


@click.command()
@click.option(
    "--input-records",
    "input_path",
    help="The file path to the input set of ESP records to select from. This "
    "should be a pickle dictionary with at minimum a `esp-records` entry.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--exclusions",
    "exclusions_path",
    help="The file path to the set of molecules (.SMI) to exclude from the output.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    "--output-records",
    "output_path",
    help="The path to save the selected records to as a pickled list of ESP "
    "record objects.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-coverage",
    "coverage_path",
    help="The path to save a coverage report of how many times each parameter is "
    "excercised by the selected ESP record set.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--params-bcc",
    "bcc_collection_path",
    help="The path to a JSON serialized BCC parameter collection.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--params-vsite",
    "vsite_collection_path",
    help="The path to a JSON serialized v-site parameter collection.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--n-processes",
    help="The number of processes to parallelize the selection across.",
    type=int,
)
def main(
    input_path,
    exclusions_path,
    output_path,
    coverage_path,
    bcc_collection_path,
    vsite_collection_path,
    n_processes,
):

    console = rich.get_console()

    vsite_collection = VirtualSiteCollection.parse_file(vsite_collection_path)
    bcc_collection = BCCCollection.parse_file(bcc_collection_path)

    # Load in the training and test sets, filtering out any redundant molecules.
    if exclusions_path is not None:

        with open(exclusions_path) as file:
            exclusions = {
                to_canonical_smiles(smiles)
                for smiles in file.read().split("\n")
                if len(smiles) > 0
            }
    else:
        exclusions = {}

    console.print(
        f"extracting records from [repr.filename]{input_path}[/repr.filename]"
    )

    with open(input_path, "rb") as file:
        esp_records = pickle.load(file)["esp-records"]

    esp_records_by_molecule = defaultdict(list)

    n_excluded = 0

    for esp_record in track(esp_records, description="sorting records by molecule"):

        canonical_smiles = to_canonical_smiles(esp_record.tagged_smiles)

        if canonical_smiles in exclusions:
            n_excluded += 1
            continue

        esp_records_by_molecule[canonical_smiles].append(esp_record)

    if len(exclusions) > 0:
        console.print(
            f"{n_excluded} molecules exclude, {len(esp_records_by_molecule)} retained"
        )

    with Pool(processes=n_processes) as pool:

        process_func = functools.partial(
            process_molecule,
            vsite_collection=vsite_collection,
            bcc_collection=bcc_collection,
        )
        processed_molecules = list(
            track(
                pool.imap(process_func, esp_records_by_molecule),
                description="filtering molecules with no v-sites",
                total=len(esp_records_by_molecule),
            )
        )

    n_without_vsite_params = 0
    n_missing_bcc_params = 0

    filtered_esp_records = []
    coverage = {"v-site": defaultdict(int), "bcc": defaultdict(int)}

    for processed_molecule in processed_molecules:

        (
            smiles,
            has_vsite_params,
            vsite_keys,
            is_missing_bcc_params,
            bcc_keys,
        ) = processed_molecule

        if not has_vsite_params:
            n_without_vsite_params += 1
            continue
        if is_missing_bcc_params:
            n_missing_bcc_params += 1
            continue

        filtered_esp_records.extend(esp_records_by_molecule[smiles])

        for vsite_key in vsite_keys:
            coverage["v-site"][vsite_key] += 1
        for bcc_key in bcc_keys:
            coverage["bcc"][bcc_key] += 1

    console.print(
        f"{n_without_vsite_params} records would not be assigned v-sites and were "
        f"removed"
    )
    console.print(
        f"{n_missing_bcc_params} records could not be fully assigned BCC parameters "
        f"and were removed"
    )

    with open(output_path, "wb") as file:
        pickle.dump(filtered_esp_records, file)
    with open(coverage_path, "w") as file:
        json.dump(coverage, file)


if __name__ == "__main__":
    main()
