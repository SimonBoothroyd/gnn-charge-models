import functools
from typing import Optional

import click
from nagl.utilities.toolkits import stream_from_file
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalize
from tqdm import tqdm


@functools.lru_cache
def canonical_smiles(smiles: str) -> Optional[str]:

    rd_molecule = Chem.MolFromSmiles(smiles)

    if rd_molecule is None:
        print(f"{smiles} could not be parsed")
        return

    rd_molecule = Normalize(rd_molecule)

    return Chem.MolToSmiles(
        Chem.AddHs(rd_molecule), isomericSmiles=False, canonical=True
    )


@click.option(
    "--input",
    "input_paths",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    default="joined.smi",
    help="The path [.smi] to save the joined files to.",
    show_default=True,
)
@click.command()
def main(input_paths, output_path):

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    joined_smiles = sorted(
        {
            canonical_smiles(smiles)
            for input_path in input_paths
            for smiles in tqdm(stream_from_file(input_path, as_smiles=True))
            if canonical_smiles(smiles) is not None
        }
    )

    with open(output_path, "w") as file:
        file.write("\n".join(joined_smiles))


if __name__ == "__main__":
    main()
