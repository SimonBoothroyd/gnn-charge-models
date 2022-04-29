import os

import rich
from nagl.storage import MoleculeStore
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from rich import pretty


def main():

    console = rich.get_console()
    pretty.install(console)

    with capture_toolkit_warnings():

        with console.status("retrieving charges"):

            charge_store = MoleculeStore(
                "../../data-set-labelling/resp-charges/industry-set.sqlite"
            )
            molecule_records = charge_store.retrieve()

        with console.status("creating library charges"):

            charge_collection = LibraryChargeCollection(
                parameters=[
                    LibraryChargeParameter(
                        smiles=molecule_record.smiles,
                        value=molecule_record.conformers[0].partial_charges[0].values,
                    )
                    for molecule_record in molecule_records
                ]
            )

    os.makedirs("reference-charges", exist_ok=True)

    with open("reference-charges/resp-charges-industry-set.json", "w") as file:
        file.write(charge_collection.json(indent=2))


if __name__ == "__main__":
    main()
