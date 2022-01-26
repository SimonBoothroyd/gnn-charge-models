import itertools

from openeye import oechem


def main():

    natural_codes = [
        # # Positively charged
        "R",
        "H",
        "K",
        # Negatively charged
        "D",
        "E",
        # Polar uncharged
        "S",
        "T",
        "N",
        "Q",
        # # Special cases
        "G",
        "P",
        "C",
        # # Hydrophobic
        "A",
        "V",
        "I",
        "L",
        "M",
        # # Hydrophobic
        "F",
        "Y",
        "W",
    ]

    smiles = []

    # Define the set of 'reactions' that will set the correct protonation state of each
    # residue.
    for codes in (
        [(code,) for code in natural_codes]
        + [*itertools.combinations(natural_codes, r=2)]
        + [(code, code) for code in natural_codes]
    ):

        oe_molecule = oechem.OEGraphMol()
        oechem.OEFastaToMol(oe_molecule, "".join(codes))
        oechem.OEAddExplicitHydrogens(oe_molecule)

        # Replace the terminal C=O group with a C(O)=O
        rxn = oechem.OEUniMolecularRxn("[#6:1]([H:2])=[#8:3]>>[#6:1]([OH1:2])=[#8:3]")
        assert rxn(oe_molecule)

        smiles.append(oechem.OEMolToSmiles(oe_molecule))

    with open("processed/esp-amino-acid-set.smi", "w") as file:
        file.write("\n".join(smiles))


if __name__ == "__main__":
    main()
