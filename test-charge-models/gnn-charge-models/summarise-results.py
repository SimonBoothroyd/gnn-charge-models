import json
import os
from typing import Dict, List

import numpy
from nagl.utilities.toolkits import capture_toolkit_warnings
from openeye import oechem, oedepict
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.toolkit.topology import Molecule
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import Draw


class LabelPartialCharges(oedepict.OEDisplayAtomPropBase):
    def __init__(self):
        oedepict.OEDisplayAtomPropBase.__init__(self)

    def __call__(self, atom: oechem.OEAtomBase):
        return f"{atom.GetPartialCharge():.3f}"

    def CreateCopy(self):
        copy = LabelPartialCharges()
        return copy.__disown__()


def draw_charges(
    molecule: Molecule, charge_sets: Dict[str, List[float]], output_path: str
):

    from openmm import unit as openmm_unit

    image_size = 1200
    image = oedepict.OEImage(image_size * len(charge_sets), image_size)

    grid = oedepict.OEImageGrid(image, 1, len(charge_sets))

    options = oedepict.OE2DMolDisplayOptions(
        grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale
    )

    atom_labeler = LabelPartialCharges()
    options.SetAtomPropertyFunctor(atom_labeler)

    for column, (label, charge_set) in enumerate(charge_sets.items()):

        molecule.partial_charges = (
            numpy.array(charge_set) * openmm_unit.elementary_charge
        )

        oe_molecule: oechem.OEMol = molecule.to_openeye()
        oe_molecule.SetTitle(label)

        oedepict.OEPrepareDepiction(oe_molecule, False, False)
        display = oedepict.OE2DMolDisplay(oe_molecule, options)

        cell = grid.GetCell(1, column + 1)
        oedepict.OERenderMolecule(cell, display)

    oedepict.OEWriteImage(output_path, image)


def main():

    with capture_toolkit_warnings():
        am1_collection = LibraryChargeCollection.parse_file(
            "../reference-charges/am1-charge-industry-set.json"
        )
        am1bcc_collection = LibraryChargeCollection.parse_file(
            "../reference-charges/am1bcc-charge-industry-set.json"
        )
        gnn_am1_collection = LibraryChargeCollection.parse_file("gnn-am1-v2.json")
        gnn_am1bcc = BCCCollection.parse_file(
            "../../train-charge-models/gnn-charge-models/gnn-am1-v2-bcc/lr-0.0025-n-400/"
            "final-parameters-bcc.json"
        )

    with open("outputs/test-per-molecule-rmse-gnn-am1bcc.json") as file:
        per_molecule_rmse: Dict[str, float] = json.load(file)

    conversion = (1.0 * unit.avogadro_constant * unit.hartree).m_as(
        unit.kilocalorie / unit.mole
    )

    ranked_molecules = sorted(
        per_molecule_rmse, key=lambda x: per_molecule_rmse[x], reverse=True
    )

    top_10_worst = ranked_molecules[:10]

    rd_molecules = [Chem.MolFromSmiles(smiles) for smiles in top_10_worst]
    Draw.MolsToGridImage(
        rd_molecules,
        5,
        (300, 300),
        [f"{per_molecule_rmse[smiles] * conversion:.4f}" for smiles in top_10_worst],
    ).save("worst-molecules.png")

    os.makedirs("worst-molecules", exist_ok=True)

    for i, worst_smiles in enumerate(top_10_worst):

        worst_molecule = Molecule.from_smiles(worst_smiles, allow_undefined_stereo=True)

        am1_charges = LibraryChargeGenerator.generate(worst_molecule, am1_collection)
        am1_charges = [float(x) for x in am1_charges]

        am1bcc_charges = LibraryChargeGenerator.generate(
            worst_molecule, am1bcc_collection
        )
        am1bcc_charges = [float(x) for x in am1bcc_charges]

        gnn_am1_charges = LibraryChargeGenerator.generate(
            worst_molecule, gnn_am1_collection
        )
        gnn_am1bcc_charges = gnn_am1_charges + BCCGenerator.generate(
            worst_molecule, gnn_am1bcc
        )

        gnn_am1_charges = [float(x) for x in gnn_am1_charges]
        gnn_am1bcc_charges = [float(x) for x in gnn_am1bcc_charges]

        print([f"{x:.4f}" for x in am1bcc_charges])
        print([f"{x:.4f}" for x in gnn_am1bcc_charges])

        draw_charges(
            worst_molecule,
            {"am1": am1_charges, "gnn-am1": gnn_am1_charges},
            f"worst-molecules/gnn-am1-v2-mol-{i}.png",
        )
        draw_charges(
            worst_molecule,
            {"am1bcc": am1bcc_charges, "gnn-am1bcc": gnn_am1bcc_charges},
            f"worst-molecules/gnn-am1-v2-bcc-mol-{i}.png",
        )


if __name__ == "__main__":
    main()
