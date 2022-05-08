import functools
import pickle
from io import BytesIO
from multiprocessing import Pool
from typing import List, Optional
from urllib.request import urlopen

import click
import numpy
import rich
import rich.console
import torch
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit.topology import Molecule
from rich import pretty
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    track,
)


@functools.lru_cache()
def strip_map_indices(smiles: str) -> str:
    return Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_smiles(
        mapped=False
    )


def download_file(url: str, description: str) -> bytes:
    """Downloads a file while showing a pretty ``rich`` progress bar."""

    with Progress(
        TextColumn(f"[bold blue]{description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=False,
    ) as progress:

        task_id = progress.add_task("download", start=False)

        with urlopen(url) as response:

            progress.update(task_id, total=int(response.info()["Content-length"]))

            with BytesIO() as buffer:

                progress.start_task(task_id)

                for data in iter(functools.partial(response.read, 32768), b""):

                    buffer.write(data)
                    progress.update(task_id, advance=len(data))

                return buffer.getvalue()


def to_library_parameter(smiles: str) -> Optional[LibraryChargeParameter]:

    import espaloma
    from simtk import unit as simtk_unit

    error_console = rich.console.Console(stderr=True)

    try:

        with capture_toolkit_warnings():

            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

            # create an Espaloma Graph object to represent the molecule of interest
            molecule_graph = espaloma.Graph(molecule)

            # apply a trained espaloma model to assign parameters
            espaloma_model = torch.load("espaloma-0.2.2.pt")
            espaloma_model(molecule_graph.heterograph)

            charge_tensor = (
                molecule_graph.nodes["n1"]
                .data["q"]
                .flatten()
                .detach()
                .cpu()
                .numpy()
                .astype(numpy.float64)
            )
            charges = [float(x) for x in charge_tensor.flatten().tolist()]

            total_charge = molecule.total_charge.value_in_unit(
                simtk_unit.elementary_charge
            )
            sum_charge = sum(charges)

            charge_difference = total_charge - sum_charge

            if not numpy.isclose(charge_difference, 0.0, atol=1.0e-4):

                error_console.print(
                    f"{smiles} - "
                    f"expected charge={total_charge}  "
                    f"actual charge={sum_charge}"
                )

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
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option("--n-processes", type=int, default=1)
def main(input_records_path, output_path, n_processes):

    console = rich.get_console()
    pretty.install(console)

    with open("espaloma-0.2.2.pt", "rb") as file:

        raw_model = download_file(
            "https://github.com/choderalab/espaloma/releases/download/"
            "0.2.2/espaloma-0.2.2.pt",
            "downloading 'espaloma-0.2.2.pt'",
        )
        file.write(raw_model)

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

            to_library_parameter_func = functools.partial(to_library_parameter)

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
