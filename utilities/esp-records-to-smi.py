import pickle
from typing import List

import click
from openff.recharge.esp.storage import MoleculeESPRecord


@click.command()
@click.option("--input", "input_path")
@click.option("--output", "output_path")
def main(input_path, output_path):

    with open(input_path, "rb") as file:
        records: List[MoleculeESPRecord] = pickle.load(file)["esp-records"]

    smiles = {record.tagged_smiles for record in records}

    with open(output_path, "w") as file:
        file.write("\n".join(smiles))


if __name__ == "__main__":
    main()
