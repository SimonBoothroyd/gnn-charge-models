import functools
import json
import pickle
from io import BytesIO
from urllib.request import urlopen

from openff.toolkit.topology import Molecule
from rdkit import Chem
from rdkit.Chem import Draw
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


def main():

    freesolv = pickle.loads(
        download_file(
            "https://github.com/MobleyLab/FreeSolv/raw/"
            "ff0961a3177863c8002e8487ff9718c513974138/database.pickle",
            "downloading FreeSolv (@ff0961a)",
        )
    )

    matches = set()

    for entry in freesolv.values():

        if "P" in entry["smiles"]:
            # OpenFF Recharge BCCs don't currently support P.
            # See https://github.com/openforcefield/openff-recharge/issues/109
            continue

        molecule: Molecule = Molecule.from_smiles(
            entry["smiles"], allow_undefined_stereo=True
        )

        smarts_matches = [
            *molecule.chemical_environment_matches(
                "[#6X3H1a:1]1:[#7X2a:2]:[#6X3H1a:3]:[#6X3a]:[#6X3a]:[#6X3a]1"
            ),
            *molecule.chemical_environment_matches("[#6:1][#17:2]"),
            *molecule.chemical_environment_matches("[#6:1][#35:2]"),
        ]

        if len(smarts_matches) == 0:
            continue

        matches.add(molecule.to_smiles(mapped=False))

    mols = [Chem.MolFromSmiles(smiles) for smiles in matches]
    Draw.MolsToGridImage(mols, 6, (300, 300)).save("test-set.png")

    with open("test-set.json", "w") as file:
        json.dump([(smiles, "O") for smiles in matches], file)


if __name__ == "__main__":
    main()
