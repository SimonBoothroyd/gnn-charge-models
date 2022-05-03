import json
import pickle

from openff.toolkit.topology import Molecule
from rdkit import Chem
from rdkit.Chem import Draw


def main():

    with open("freesolv.pkl", "rb") as file:
        freesolv = pickle.load(file)

    matches = set()

    for entry in freesolv.values():

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
