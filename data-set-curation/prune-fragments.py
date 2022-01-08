import click
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs import DiceSimilarity
from rdkit.SimDivFilters import MaxMinPicker


@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="The path to the file [.smi] containing the list of fragments to prune.",
    required=True,
)
@click.option(
    "--n-fragments",
    type=int,
    default=20000,
    help="The number of diverse fragments to select.",
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    default="pruned.smi",
    help="The path to the file [.smi] to save the selected fragments to.",
    show_default=True,
)
@click.command()
def main(input_path, n_fragments, output_path):

    with open(input_path) as file:
        fragment_smiles = [x for x in file.read().split("\n") if len(x) > 0]

    fragments = [
        Chem.MolFromSmiles(fragment_pattern)
        for fragment_pattern in fragment_smiles
        if len(fragment_pattern) > 0
    ]
    fingerprints = [
        GetMorganFingerprint(fragment, 3) for fragment in fragments if fragment
    ]

    def dice_distance(i, j):
        return 1.0 - DiceSimilarity(fingerprints[i], fingerprints[j])

    picker = MaxMinPicker()

    selected_indices = picker.LazyPick(
        dice_distance, len(fingerprints), n_fragments, seed=42
    )
    selected_smiles = [fragment_smiles[i] for i in selected_indices]

    with open(output_path, "w") as file:
        file.write("\n".join(selected_smiles))


if __name__ == "__main__":
    main()
