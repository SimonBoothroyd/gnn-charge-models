import os
from collections import defaultdict
from pprint import pprint
from typing import List

import click
from constructure.constructors import OpenEyeConstructor
from constructure.scaffolds import Scaffold
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator, BCCParameter
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.toolkit.topology import Molecule
from pydantic import parse_file_as


@click.command()
def main():

    reactions = [
        # ------------------ P valence 3 ------------------
        # === [R]-P(-[R])-[R]
        (
            "[P]([R1])([R2])([R3])",
            {
                1: [
                    "[R][H]",
                    "[R]C",
                    "[R]C=C",
                    "[R]C#C",
                    "[R]C=N",
                    "[R]C=P",
                    "[R]C(=O)C",
                    "[R]C(=O)O",
                    "[R]C(=O)OC",
                    "[R]C(=S)C",
                    "[R]C(=S)S",
                    "[R]C(=S)SC",
                    "[R]N",
                    "[R]NC",
                    "[R]N(C)C",
                    "[R]N=C",
                    "[R]NC(=O)",
                    "[R]NC(=O)C",
                    "[R]O",
                    "[R]OC",
                    "[R]S",
                    "[R]SC",
                    "[R]P",
                    "[R]PC",
                    "[R]P(C)C",
                    "[R]P(C)(C)=O",
                    "[R]P(O)(O)=O",
                    "[R]P(OC)(OC)=O",
                    "[R]P(C)(C)=S",
                    "[R]P(O)(O)=S",
                    "[R]P(OC)(OC)=S",
                    "[R]C1=CC=CC=C1",
                    "[R]C1=CC=CC=N1",
                    "[R]F",
                    "[R]Cl",
                    "[R]Br",
                    "[R]I",
                ],
                2: ["[R][H]", "[R]C"],
                3: ["[R][H]", "[R]C"],
            },
        ),
        # === [R]-P=[R]
        (
            "[P]([R1])([R2])",
            {
                1: ["[R]=C", "[R]=C=C", "[R]=O", "[R]=N", "[R]=NC"],
                2: ["[R][H]", "[R]C", "[R]N", "[R]NC", "[R]N(C)C", "[R]O", "[R]OC"],
            },
        ),
        (
            "[P]([R1])=[P]([R2])",
            {
                1: ["[R][H]", "[R]C"],
                2: ["[R][H]", "[R]C"],
            },
        ),
        # === C-P-[x] 5 membered ring
        *[
            (
                f"{x}1=CC([R2])=C([R3])P1([R1])",
                {
                    1: ["[R][H]", "[R]C", "[R][O-]", "[R]O", "[R][N+](=O)([O-])"],
                    2: ["[R][H]", "[R]C"],
                    3: ["[R][H]", "[R]C"],
                },
            )
            for x in ["C", "N"]
        ],
        # === C=P-[x] 5 membered ring
        *[
            (f"{x}1C=C([R1])C([R1])=P1", {1: ["[R][H]", "[R]C"], 2: ["[R][H]", "[R]C"]})
            for x in ["C", "N", "O", "S"]
        ],
        # === C-P=[x] 6 membered ring
        *[
            (
                f"C1=CC=CC={x}1P([R1])([R2])",
                {1: ["[R][H]", "[R]C", "[R][N+](=O)([O-])"], 2: ["[R][H]", "[R]C"]},
            )
            for x in ["C"]
        ],
        # ------------------ valence 5 ------------------
        (
            "[P]([R1])([R2])([R3])([R4])",
            {
                1: [
                    "[R][H]",
                    "[R]C",
                    "[R]C#C",
                    "[R]O",
                    "[R]OC",
                    "[R]N",
                    "[R]NC",
                    "[R]N(C)C",
                    "[R]NC(=O)C",
                    "[R][N+H3]",
                    "[R][N+H2]C",
                    "[R][N+H](C)C",
                    "[R]C1=CC=CC=C1",
                    "[R]C1=CC=CC=N1",
                    "[R]N1C=CC=C1",
                    "[R]S(=O)C",
                    "[R]S(=O)CC",
                    "[R]P(=O)(O)(O)",
                    "[R]F",
                    "[R]Br",
                    "[R]I",
                ],
                2: ["[R][H]", "[R]C", "[R]O", "[R]OC", "[R]N", "[R]NC", "[R]N(C)C"],
                3: ["[R][H]", "[R]C", "[R]O", "[R]OC", "[R]N", "[R]NC", "[R]N(C)C"],
                4: ["[R]=O"],
            },
        ),
        # === 150242
        (
            "[P]([R1])([R2])([R3])([R4])",
            {
                1: ["[R][H]", "[R]C"],
                2: ["[R][H]", "[R]C"],
                3: ["[R][H]", "[R]C"],
                4: ["[R]=C=C", "[R]=C=CC", "[R]=C=C(C)C"],
            },
        ),
        # ------------------ halogens ------------------
        # "140171",
        # "140173",
        # "140174",
        ("C([R1])([R2])=O", {1: ["[R][H]", "[R]C"], 2: ["[R]F", "[R]Br", "[R]I"]}),
        # "150171",
        # "150172",
        ("C#C([R1])", {1: ["[R]F", "[R]Cl"]}),
        ("CC#C([R1])", {1: ["[R]F", "[R]Cl"]}),
        ("CCC#C([R1])", {1: ["[R]F", "[R]Cl"]}),
        # "210171",
        # "210172",
        # "210173",
        # "210174",
        (
            "N([R1])([R2])([R3])",
            {
                1: ["[R][H]", "[R]C"],
                2: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
                3: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
            },
        ),
        # "220171",
        # "220172",
        # "220173",
        # "220174",
        (
            "N([R1])([R2])C(=O)([R3])",
            {
                1: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
                2: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
                3: ["[R][H]", "[R]C"],
            },
        ),
        # "230171",
        # "230172",
        # "230173",
        (
            "[N+]([R1])([R2])([R3])",
            {
                1: ["[R]=C", "[R]=CC"],
                2: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
                3: ["[R][H]", "[R]C", "[R]F", "[R]Cl", "[R]Br", "[R]I"],
            },
        ),
        # "240171",
        # "240172",
        # "240173",
        # "240174",
        ("C=N([R1])", {1: ["[R]F", "[R]Cl", "[R]Br", "[R]I"]}),
        ("CC=N([R1])", {1: ["[R]F", "[R]Cl", "[R]Br", "[R]I"]}),
        # "310172",
        # "310173",
        # "310174",
        ("CO([R1])", {1: ["[R]Cl", "[R]Br", "[R]I"]}),
        ("CCO([R1])", {1: ["[R]Cl", "[R]Br", "[R]I"]}),
        ("C(C)CO([R1])", {1: ["[R]Cl", "[R]Br", "[R]I"]}),
        # "510171",
        # "510172",
        # "510173",
        ("CS([R1])", {1: ["[R]F", "[R]Cl", "[R]Br"]}),
        ("CCS([R1])", {1: ["[R]F", "[R]Cl", "[R]Br"]}),
        ("C(C)CS([R1])", {1: ["[R]F", "[R]Cl", "[R]Br"]}),
        # "520172",
        ("S(=O)([R1])([R2])", {1: ["[R][H]", "[R]C", "[R]CC", "[R]Cl"], 2: ["[R]Cl"]}),
        # "530173",
        # "530174",
        (
            "S(=O)(=O)([R1])([R2])",
            {1: ["[R]Br", "[R]I"], 2: ["[R]C", "[R]CC", "[R]CCC"]},
        ),
        # ------------carbon + XXX-------------
        # "120125",
        (
            "N#[N+]C([R1])=C([R2])",
            {1: ["[R][H]", "[R]C", "[R]CC"], 2: ["[R][H]", "[R]C"]},
        ),
        (
            "C#[N+]C([R1])=C([R2])",
            {1: ["[R][H]", "[R]C", "[R]CC"], 2: ["[R][H]", "[R]C"]},
        ),
        # "140125",
        ("N#[N+]C(=O)C([R1])", {1: ["[R][H]", "[R]C", "[R]CC"]}),
        ("C#[N+]C(=O)C([R1])", {1: ["[R][H]", "[R]C", "[R]CC"]}),
        # "140232",
        # Huh?
        # "150215",
        # "150231",
        # "150252",
        (
            "[C]([R1])=[C]([R2])",
            {
                1: ["[R]=C", "[R]=CC"],
                2: ["[R]=C", "[R]=CC", "[R]=O", "[R]=S", "[R]=N", "[R]=NC"],
            },
        ),
        # "150122",
        ("N#CNC(=O)([R1])", {1: ["[R]C", "[R]CC"]}),
        ("C#CNC(=O)([R1])", {1: ["[R]C", "[R]CC"]}),
        # "150125",
        ("N#C[N+]([R1])", {1: ["[R]#C", "[R]#CC", "[R]#N"]}),
        ("C#C[N+]([R1])", {1: ["[R]#C", "[R]#CC", "[R]#N"]}),
        # "150152",
        ("C#CS(=O)([R1])", {1: ["[R][H]", "[R]C", "[R]CC", "[R]C(C)C"]}),
        # "150153",
        ("C#CS(=O)(=O)([R1])", {1: ["[R]C", "[R]CC", "[R]C(C)C", "[R]N"]}),
        # "160731",
        # Huh?
        # "170125",
        # "170951",
        ("C1=CC=C(O1)([R1])", {1: ["[R][N+]#C", "[R][N+]#CC", "[R][S-]"]}),
        ("C1=CC=C(N1)([R1])", {1: ["[R][N+]#C", "[R][N+]#CC", "[R][S-]"]}),
        # -----------nitrogen + XXX------------
        # "220125",
        # "220631",
        # "240125",
        # "240251",
        # "250631",
        # "250651",
        # ------------oxygen + XXX-------------
        # "310251",
        ("O=S=C([R1])([R2])", {1: ["[R][H]", "[R]C", "[R]CC"], 2: ["[R][H]", "[R]C"]}),
        # "310751",
        # ------------sulfur + XXX-------------
        # "510252",
        # "520152",
        # --------------hydrogen---------------
        # "250191",
        # -----------dative / charged----------
        # "110931",
        (
            "[O-]-C([R1])([R2])([R3])",
            {
                1: ["[R][H]", "[R]C", "[R]CC"],
                2: ["[R][H]", "[R]C", "[R]CC"],
                3: ["[R][H]", "[R]C", "[R]CC"],
            },
        ),
        # "110951",
        (
            "[S-]-C([R1])([R2])([R3])",
            {
                1: ["[R][H]", "[R]C", "[R]CC"],
                2: ["[R][H]", "[R]C", "[R]CC"],
                3: ["[R][H]", "[R]C", "[R]CC"],
            },
        ),
        # "130951",
        (
            "[S-]C(=N([R1]))([R2])",
            {1: ["[R][H]", "[R]C", "[R]CC"], 2: ["[R][H]", "[R]C", "[R]CC"]},
        ),
        # "150931",
        ("[O-][C]([R1])", {1: ["[R]#N", "[R]#C", "[R]#CC", "[R]#CCC", "[R]#CC(C)C"]}),
        # "150951",
        ("[S-][C]([R1])", {1: ["[R]#N", "[R]#C", "[R]#CC", "[R]#CCC", "[R]#CC(C)C"]}),
        # "310941",
        # "310942",
        # "310951",
        # "510953"
    ]

    products = set()

    for scaffold_smiles, substituents in reactions:

        scaffold = Scaffold(
            smiles=scaffold_smiles,
            r_groups={
                i: ["hydrogen", "acyl", "alkyl", "aryl", "halogen", "hetero", None]
                for i in substituents
            },
        )

        products.update(
            OpenEyeConstructor.enumerate_combinations(
                scaffold, substituents=substituents
            )
        )

    am1bcc = BCCCollection(parameters=parse_file_as(List[BCCParameter], "am1bcc.json"))

    valid_smiles = set()
    coverage = defaultdict(int)

    for pattern in products:

        molecule: Molecule = Molecule.from_smiles(pattern, allow_undefined_stereo=True)

        try:
            applied = BCCGenerator.applied_corrections(molecule, bcc_collection=am1bcc)
        except ChargeAssignmentError:
            print(f"skipping {pattern} - cannot be assigned BCC parameters")
            continue

        for bcc in applied:
            coverage[bcc.provenance["code"]] += 1

        valid_smiles.add(pattern)

    with open(os.path.join("data", "processed", "esp-am1bcc-set.smi"), "w") as file:
        file.write("\n".join(valid_smiles))

    pprint(coverage)


if __name__ == "__main__":
    main()
