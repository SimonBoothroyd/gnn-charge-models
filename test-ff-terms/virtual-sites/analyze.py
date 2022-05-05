import functools
import json
import pickle
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import click
import numpy
import pandas
import rich
import seaborn
from absolv.models import TransferFreeEnergyResult
from matplotlib import pyplot
from openff.toolkit.topology import Molecule
from openff.units import unit
from openff.units.openmm import from_openmm
from openff.utilities import temporary_cd
from rdkit import Chem
from rdkit.Chem import Draw
from rich import pretty
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
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


def load_reference() -> Tuple[
    Dict[int, Tuple[unit.Quantity, unit.Quantity]], Dict[str, int]
]:

    with open("test-set.json") as file:
        test_set = json.load(file)

    test_set_smiles = {
        Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_smiles(): i + 1
        for i, (smiles, _) in enumerate(test_set)
    }

    freesolv = pickle.loads(
        download_file(
            "https://github.com/MobleyLab/FreeSolv/raw/"
            "ff0961a3177863c8002e8487ff9718c513974138/database.pickle",
            "downloading FreeSolv (@ff0961a)",
        )
    )

    reference_properties = {}

    for entry in freesolv.values():

        entry_smiles = Molecule.from_smiles(
            entry["smiles"], allow_undefined_stereo=True
        ).to_smiles()

        if entry_smiles not in test_set_smiles:
            continue

        entry_id = test_set_smiles[entry_smiles]
        reference_properties[entry_id] = (
            entry["expt"] * unit.kilocalorie / unit.mole,
            entry["d_expt"] * unit.kilocalorie / unit.mole,
        )

    return reference_properties, test_set_smiles


def load_results_results(
    name: str, smiles_to_id: Dict[str, int]
) -> Dict[int, Tuple[unit.Quantity, unit.Quantity]]:

    result_properties = {}

    for result_file in Path(f"results/{name}/").glob("*.json"):

        result = TransferFreeEnergyResult.parse_file(result_file)

        smiles = Molecule.from_smiles(
            next(iter(result.input_schema.system.solutes)), allow_undefined_stereo=True
        ).to_smiles()

        if "P" in smiles:
            continue

        value, uncertainty = result.delta_g_from_a_to_b_with_units

        result_properties[smiles_to_id[smiles]] = (
            from_openmm(value),
            from_openmm(uncertainty),
        )

    return result_properties


@click.command()
def main():

    console = rich.get_console()
    pretty.install(console)

    reference_data_set, smiles_to_id = load_reference()
    id_to_smiles = {id: smiles for smiles, id in smiles_to_id.items()}

    force_field_0 = "openff-2-0-0"
    force_field_1 = "vam1bcc-v1"

    results_per_force_field = {
        "openff-2-0-0": load_results_results("openff-2-0-0", smiles_to_id),
        "vam1bcc-v1": load_results_results("vam1bcc-v1", smiles_to_id),
    }

    finished_ids = set.intersection(
        *[
            set(entry_id for entry_id in results)
            for results in results_per_force_field.values()
        ]
    )

    missing_ids = {*reference_data_set} - finished_ids

    if len(missing_ids) > 0:
        console.print("results missing for:", missing_ids)

    deltas = []

    smiles_to_delta = {}

    label = r"$\Delta\Delta\Delta G_{vsite-orig}$ (kcal / mol)"

    for entry_id in finished_ids:

        ff_value_0 = results_per_force_field[force_field_0][entry_id][0]
        ff_value_1 = results_per_force_field[force_field_1][entry_id][0]

        ref_value = reference_data_set[entry_id][0]

        delta = float(
            (
                numpy.abs(ff_value_1 - ref_value) - numpy.abs(ff_value_0 - ref_value)
            ).m_as(unit.kilocalorie / unit.mole)
        )

        deltas.append({"ID": entry_id, label: delta})

        smiles_to_delta[id_to_smiles[entry_id]] = delta

    plot_data = pandas.DataFrame(deltas)

    output_directory = Path("analysis")
    output_directory.mkdir(parents=True, exist_ok=True)

    with temporary_cd(str(output_directory)):

        seaborn.histplot(plot_data, x=label)
        pyplot.savefig("dddG.png")
        pyplot.show()

        sorted_smiles = sorted(smiles_to_delta, key=lambda x: smiles_to_delta[x])

        molecules = [Chem.MolFromSmiles(smiles) for smiles in sorted_smiles]
        labels = [f"{smiles_to_delta[smiles]:.4f}" for smiles in sorted_smiles]

        Draw.MolsToGridImage(molecules, 10, (300, 300), labels).save(
            "per-mol-dddG-kcal-mol.png"
        )


if __name__ == "__main__":
    main()
