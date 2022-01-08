import logging
import os
from collections import defaultdict

import click
from matplotlib import pyplot
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Recap
from tqdm import tqdm


def canonical_smiles(rd_molecule: Chem.Mol) -> str:

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
    "output_directory",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="fragments",
    help="The directory to the save the generated fragments to.",
    show_default=True,
)
@click.command()
def main(input_paths, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    # Load in the molecules to fragment
    with capture_toolkit_warnings():

        all_parent_smiles = sorted(
            {
                smiles
                for input_path in input_paths
                for smiles in tqdm(stream_from_file(input_path, as_smiles=True))
            }
        )

    # Fragment the molecules
    allowed_elements = {"B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si"}

    rd_dummy = Chem.MolFromSmiles("*")
    rd_hydrogen = Chem.MolFromSmiles("[H]")

    unique_parents_by_n_heavy = defaultdict(set)
    unique_fragments_by_n_heavy = defaultdict(set)

    for parent_smiles in tqdm(all_parent_smiles):

        rd_parent: Chem.Mol = Chem.MolFromSmiles(parent_smiles)

        if any(
            rd_atom.GetNumRadicalElectrons() != 0
            or rd_atom.GetIsotope() != 0
            or rd_atom.GetSymbol() not in allowed_elements
            for rd_atom in rd_parent.GetAtoms()
        ):
            continue

        unique_parents_by_n_heavy[rd_parent.GetNumHeavyAtoms()].add(
            canonical_smiles(rd_parent)
        )

        fragment_nodes = Recap.RecapDecompose(rd_parent).GetLeaves()

        for fragment_node in fragment_nodes.values():

            rd_fragment = AllChem.ReplaceSubstructs(
                fragment_node.mol, rd_dummy, rd_hydrogen, True
            )[0]
            # Do a SMILES round-trip to avoid wierd issues with radical formation...
            rd_fragment = Chem.MolFromSmiles(Chem.MolToSmiles(rd_fragment))

            if Descriptors.NumRadicalElectrons(rd_fragment) > 0:
                logging.warning(f"A fragment of {parent_smiles} has a radical electron")
                continue

            fragment_smiles = canonical_smiles(rd_fragment)

            if (
                fragment_smiles
                in unique_fragments_by_n_heavy[rd_fragment.GetNumHeavyAtoms()]
            ):
                continue

            unique_fragments_by_n_heavy[rd_fragment.GetNumHeavyAtoms()].add(
                fragment_smiles
            )

    # # Plot the distribution of parent / fragments by N heavy atoms
    # x = sorted(unique_parents_by_n_heavy)
    # y = [len(unique_parents_by_n_heavy[n]) for n in x]
    #
    # pyplot.bar(x, y)
    # pyplot.title("n_parent")
    # pyplot.show()
    #
    # x = sorted(unique_fragments_by_n_heavy)
    # y = [len(unique_fragments_by_n_heavy[n]) for n in x]
    #
    # pyplot.bar(x, y)
    # pyplot.title("n_fragment")
    # pyplot.show()

    # Save the fragments
    for n_heavy, fragment_smiles in unique_fragments_by_n_heavy.items():

        with open(
            os.path.join(output_directory, f"fragments-{n_heavy}.smi"), "w"
        ) as file:
            file.write("\n".join(fragment_smiles))

    with open(os.path.join(output_directory, f"fragments-full.smi"), "w") as file:
        file.write(
            "\n".join(
                fragment_pattern
                for n_heavy, fragment_smiles in unique_fragments_by_n_heavy.items()
                for fragment_pattern in fragment_smiles
            )
        )

    small_fragments = [
        fragment_pattern
        for n_heavy in range(3, 13)
        for fragment_pattern in unique_fragments_by_n_heavy[n_heavy]
    ]

    with open(os.path.join(output_directory, f"fragments-small.smi"), "w") as file:
        file.write("\n".join(small_fragments))


if __name__ == "__main__":
    main()
