import functools
import pickle
from multiprocessing import Pool
from typing import List, Literal, Optional

import click
import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit.topology import Molecule
from rich import pretty
from rich.progress import track


@functools.lru_cache()
def strip_map_indices(smiles: str) -> str:
    return Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_smiles(
        mapped=False
    )


def to_library_parameter(
    smiles: str, charge_method: Literal["am1", "am1bcc", "GFN1-xTB", "GFN2-xTB"]
) -> Optional[LibraryChargeParameter]:

    error_console = rich.Console(stderr=True)

    try:
        console = rich.get_console()
        console.print(f"generating charges for {smiles}")

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        conformers = ConformerGenerator.generate(
            molecule, ConformerSettings(max_conformers=10)
        )
        charges = QCChargeGenerator.generate(
            molecule, conformers, QCChargeSettings(theory=charge_method)
        )

        smiles = molecule.to_smiles(mapped=True)

        return LibraryChargeParameter(
            smiles=smiles, value=[float(x) for x in charges.flatten().tolist()]
        )

    except BaseException:
        error_console.print_exception()
        return None


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--method",
    "charge_method",
    type=click.Choice(("am1", "am1bcc", "GFN1-xTB", "GFN2-xTB"), case_sensitive=True),
    required=True,
)
@click.option("--n-processes", type=int, default=1)
def main(input_path, output_path, charge_method, n_processes):

    console = rich.get_console()
    pretty.install(console)

    with capture_toolkit_warnings():

        with open(input_path, "rb") as file:
            with console.status("loading ESP records"):
                esp_records_test: List[MoleculeESPRecord] = pickle.load(file)[
                    "esp-records"
                ]
            console.print(f"loaded {len(esp_records_test)} records")

        smiles = sorted(
            {strip_map_indices(record.tagged_smiles) for record in esp_records_test}
        )

        console.print(f"found {len(smiles)} unique SMILES")

        with Pool(processes=n_processes) as pool:

            to_library_parameter_func = functools.partial(
                to_library_parameter, charge_method=charge_method
            )

            parameters = list(
                track(
                    pool.imap(to_library_parameter_func, smiles),
                    description="charging",
                    total=len(smiles),
                )
            )

        parameters = [parameter for parameter in parameters if parameter is not None]
        charge_collection = LibraryChargeCollection(parameters=parameters)

    with open(output_path, "w") as file:
        file.write(charge_collection.json(indent=2))


if __name__ == "__main__":
    main()
