import json


def save_features(atom_feature_args, bond_feature_args, output_path: str):

    with open(output_path, "w") as file:
        json.dump((atom_feature_args, bond_feature_args), file)


def features_v1():
    atom_feature_args = [
        ("AtomicElement", (["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"],)),
        ("AtomConnectivity", ()),
        ("AtomAverageFormalCharge", ()),
        ("AtomIsInRing", ()),
    ]
    bond_feature_args = [
        ("BondIsInRing", ()),
    ]

    return atom_feature_args, bond_feature_args


def features_v2():

    atom_feature_args = [
        ("AtomicElement", (["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"],)),
        ("AtomConnectivity", ()),
        ("AtomAverageFormalCharge", ()),
        ("AtomInRingOfSize", (3,)),
        ("AtomInRingOfSize", (4,)),
        ("AtomInRingOfSize", (5,)),
        ("AtomInRingOfSize", (6,)),
    ]
    bond_feature_args = [
        ("BondInRingOfSize", (3,)),
        ("BondInRingOfSize", (4,)),
        ("BondInRingOfSize", (5,)),
        ("BondInRingOfSize", (6,)),
    ]

    return atom_feature_args, bond_feature_args


def main():

    save_features(*features_v1(), "gnn-am1-features-v1.json")
    save_features(*features_v2(), "gnn-am1-features-v2.json")


if __name__ == "__main__":
    main()
