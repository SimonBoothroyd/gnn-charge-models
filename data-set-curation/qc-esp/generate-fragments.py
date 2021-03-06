import logging
import os
from collections import defaultdict

import click
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

    rd_dummy_replacements = [
        # Handle the special case of -S(=O)(=O)[*] -> -S(=O)(-[O-])
        (Chem.MolFromSmiles("S(=O)(=O)*"), Chem.MolFromSmiles("S(=O)([O-])")),
        # Handle the general case
        (Chem.MolFromSmiles("*"), Chem.MolFromSmiles("[H]")),
    ]

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

            rd_fragment = fragment_node.mol

            for rd_dummy, rd_replacement in rd_dummy_replacements:

                rd_fragment = AllChem.ReplaceSubstructs(
                    rd_fragment, rd_dummy, rd_replacement, True
                )[0]
                # Do a SMILES round-trip to avoid wierd issues with radical formation...
                rd_fragment = Chem.MolFromSmiles(Chem.MolToSmiles(rd_fragment))

            if Descriptors.NumRadicalElectrons(rd_fragment) > 0:
                logging.warning(f"A fragment of {parent_smiles} has a radical electron")
                continue

            fragment_smiles = canonical_smiles(rd_fragment)

            if "." in fragment_smiles:
                # Skip dimers, trimers, etc.
                continue

            if (
                fragment_smiles
                in unique_fragments_by_n_heavy[rd_fragment.GetNumHeavyAtoms()]
            ):
                continue

            unique_fragments_by_n_heavy[rd_fragment.GetNumHeavyAtoms()].add(
                fragment_smiles
            )

    # Save the fragments
    with open(os.path.join(output_directory, "fragments-full.smi"), "w") as file:
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

    with open(os.path.join(output_directory, "fragments-small.smi"), "w") as file:
        file.write("\n".join(small_fragments))


if __name__ == "__main__":
    main()
