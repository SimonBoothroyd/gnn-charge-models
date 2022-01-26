import pickle

import click
import rich
from qcportal import FractalClient
from qcportal.collections import Dataset
from qcportal.models import ResultRecord


@click.option(
    "--qcf-dataset",
    "dataset_name",
    help="The name of the single point dataset to retrieve the records from.",
    type=str,
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path (.pkl) to the save the records to.",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
)
@click.command()
def main(dataset_name, output_path):

    console = rich.get_console()

    client = FractalClient()

    with console.status("loading dataset metadata"):
        dataset = Dataset.from_server(client, dataset_name)

    with console.status("downloading records"):

        records_by_id = {
            record.id: record
            for record in dataset.get_records("hf", "6-31G*")["record"]
            if isinstance(record, ResultRecord)
        }
        record_id_to_molecule_id = {
            record.id: record.molecule for record in records_by_id.values()
        }

    console.print("records downloaded")

    with console.status("downloading molecules"):

        molecule_ids = sorted(set(record_id_to_molecule_id.values()))
        molecule_ids_batched = [
            molecule_ids[i : i + client.query_limit]
            for i in range(0, len(molecule_ids), client.query_limit)
        ]
        molecules = {
            molecule.id: molecule
            for molecule_id_batch in molecule_ids_batched
            for molecule in client.query_molecules(molecule_id_batch)
        }

    console.print("molecules downloaded")

    with console.status("downloading keywords"):

        keyword_ids = list({record.keywords for record in records_by_id.values()})
        keywords = {
            keyword_id: client.query_keywords(keyword_id)[0]
            for keyword_id in keyword_ids
        }

    console.print("keywords downloaded")

    results = [
        (records_by_id[record_id], molecules[record_id_to_molecule_id[record_id]])
        for record_id in records_by_id
    ]

    with open(output_path, "wb") as file:
        pickle.dump((results, keywords), file)


if __name__ == "__main__":
    main()
