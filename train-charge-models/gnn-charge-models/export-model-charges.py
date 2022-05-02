import functools
import pickle
from multiprocessing import Pool
from typing import List, Optional

import click
import numpy
import rich
import rich.console
import torch
from models import PartialChargeModelV1
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
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
    smiles: str, model_path: str
) -> Optional[LibraryChargeParameter]:

    from simtk import unit as simtk_unit

    error_console = rich.console.Console(stderr=True)

    try:

        with capture_toolkit_warnings():

            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            model = PartialChargeModelV1.load_from_checkpoint(
                model_path, map_location=torch.device("cpu")
            )
            charge_tensor = model.compute_charges(molecule).detach().numpy()
            charges = [float(x) for x in charge_tensor.flatten().tolist()]

            total_charge = molecule.total_charge.value_in_unit(
                simtk_unit.elementary_charge
            )
            sum_charge = sum(charges)

            charge_difference = total_charge - sum_charge
            assert numpy.isclose(charge_difference, 0.0, atol=1.0e-4)

            # Fix the summed charge not being within a strict precision of the total
            # charge
            charges = [
                charge + charge_difference / molecule.n_atoms for charge in charges
            ]

            smiles = molecule.to_smiles(mapped=True)

            return LibraryChargeParameter(smiles=smiles, value=charges)

    except BaseException:

        error_console.print(f"failed generating charges for {smiles}")
        error_console.print_exception()
        return None


@click.command()
@click.option(
    "--input-records",
    "input_records_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-checkpoint",
    "model_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option("--n-processes", type=int, default=1)
def main(input_records_path, model_path, output_path, n_processes):

    console = rich.get_console()
    pretty.install(console)

    with capture_toolkit_warnings():

        with open(input_records_path, "rb") as file:
            with console.status("loading ESP records"):
                esp_records_test: List[MoleculeESPRecord] = pickle.load(file)[
                    "esp-records"
                ]
            console.print(f"loaded {len(esp_records_test)} records")

        smiles = sorted(
            {
                strip_map_indices(record.tagged_smiles)
                for record in track(
                    esp_records_test, description="finding unique SMILES"
                )
            }
        )

        console.print(f"found {len(smiles)} unique SMILES")

        with Pool(processes=n_processes) as pool:

            to_library_parameter_func = functools.partial(
                to_library_parameter, model_path=model_path
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
