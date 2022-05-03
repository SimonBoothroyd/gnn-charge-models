from typing import Any, List, Literal, Tuple

import torch
from nagl.features import (
    AtomConnectivity,
    AtomFeature,
    AtomicElement,
    AtomIsInRing,
    BondFeature,
    BondIsInRing,
)
from nagl.lightning import DGLMoleculeLightningModel
from nagl.molecules import DGLMolecule
from nagl.nn import SequentialLayers
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.postprocess import ComputePartialCharges
from nagl.resonance import enumerate_resonance_forms
from nagl.utilities.toolkits import normalize_molecule
from openeye import oechem
from openff.toolkit.topology import Molecule

AtomFeatures = Literal[
    "AtomicElement",
    "AtomConnectivity",
    "AtomAverageFormalCharge",
    "AtomIsInRing",
    "AtomInRingOfSize",
]
BondFeatures = Literal["BondIsInRing", "BondInRingOfSize"]


class AtomAverageFormalCharge(AtomFeature):
    """Computes the average formal charge on each atom in a molecule across resonance
    structures."""

    def __call__(self, molecule: Molecule) -> torch.Tensor:

        try:
            molecule = normalize_molecule(molecule)
        except AssertionError:
            # See openff-toolkit/issues/1181
            pass

        resonance_forms = enumerate_resonance_forms(
            molecule,
            as_dicts=True,
            # exclude for e.g. the charged resonance form of an amide
            lowest_energy_only=True,
            # exclude resonance structures that only differ in things like kekule
            # form
            include_all_transfer_pathways=False,
        )

        formal_charges = [
            [
                atom["formal_charge"]
                for resonance_form in resonance_forms
                if i in resonance_form["atoms"]
                for atom in resonance_form["atoms"][i]
            ]
            for i in range(molecule.n_atoms)
        ]

        feature_tensor = torch.tensor(
            [
                [
                    sum(formal_charges[i]) / len(formal_charges[i])
                    if len(formal_charges[i]) > 0
                    else 0.0
                ]
                for i in range(molecule.n_atoms)
            ]
        )

        return feature_tensor

    def __len__(self):
        return 1


class AtomInRingOfSize(AtomFeature):
    def __init__(self, ring_size: int):

        assert ring_size >= 3
        self.ring_size = ring_size

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        oe_molecule: oechem.OEMol = molecule.to_openeye()
        oechem.OEFindRingAtomsAndBonds(oe_molecule)

        return torch.tensor(
            [
                int(oechem.OEAtomIsInRingSize(oe_atom, self.ring_size))
                for oe_atom in oe_molecule.GetAtoms()
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class BondInRingOfSize(AtomFeature):
    def __init__(self, ring_size: int):
        assert ring_size >= 3
        self.ring_size = ring_size

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        oe_molecule: oechem.OEMol = molecule.to_openeye()
        oechem.OEFindRingAtomsAndBonds(oe_molecule)

        oe_bonds_by_index = {
            tuple(sorted((oe_bond.GetBgnIdx(), oe_bond.GetEndIdx()))): oe_bond
            for oe_bond in oe_molecule.GetBonds()
        }
        oe_bonds = [
            oe_bonds_by_index[tuple(sorted((bond.atom1_index, bond.atom2_index)))]
            for bond in molecule.bonds
        ]

        return torch.tensor(
            [
                int(oechem.OEBondIsInRingSize(oe_bond, self.ring_size))
                for oe_bond in oe_bonds
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class PartialChargeModelV1(DGLMoleculeLightningModel):
    def features(
        self,
    ) -> Tuple[List[AtomFeature], List[BondFeature]]:

        atom_feature_name_to_feature = {
            "AtomicElement": AtomicElement,
            "AtomConnectivity": AtomConnectivity,
            "AtomAverageFormalCharge": AtomAverageFormalCharge,
            "AtomIsInRing": AtomIsInRing,
            "AtomInRingOfSize": AtomInRingOfSize,
        }
        bond_feature_name_to_feature = {
            "BondIsInRing": BondIsInRing,
            "BondInRingOfSize": BondInRingOfSize,
        }

        atom_features = [
            atom_feature_name_to_feature[name](*args)
            for name, args in self.atom_feature_args
        ]
        bond_features = [
            bond_feature_name_to_feature[name](*args)
            for name, args in self.bond_feature_args
        ]

        return atom_features, bond_features

    def __init__(
        self,
        n_gcn_hidden_features: int,
        n_gcn_layers: int,
        n_am1_hidden_features: int,
        n_am1_layers: int,
        learning_rate: float,
        partial_charge_method: str,
        atom_features: List[Tuple[AtomFeatures, Tuple[Any, ...]]],
        bond_features: List[Tuple[BondFeatures, Tuple[Any, ...]]],
    ):

        self.n_gcn_hidden_features = n_gcn_hidden_features
        self.n_gcn_layers = n_gcn_layers

        self.n_am1_hidden_features = n_am1_hidden_features
        self.n_am1_layers = n_am1_layers

        atom_features = [
            (value, ()) if isinstance(value, str) else value
            for value in atom_features
        ]
        bond_features = [
            (value, ()) if isinstance(value, str) else value
            for value in bond_features
        ]

        self.atom_feature_args: List[
            Tuple[AtomFeatures, Tuple[Any, ...]]
        ] = atom_features
        self.bond_feature_args: List[
            Tuple[BondFeatures, Tuple[Any, ...]]
        ] = bond_features

        self.partial_charge_method = partial_charge_method

        self._charge_readout = f"{partial_charge_method}-charges"

        n_atom_features = sum(len(feature) for feature in self.features()[0])

        super(PartialChargeModelV1, self).__init__(
            convolution_module=ConvolutionModule(
                architecture="SAGEConv",
                in_feats=n_atom_features,
                hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
            ),
            readout_modules={
                self._charge_readout: ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers(
                        in_feats=n_gcn_hidden_features,
                        hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                        activation=["ReLU"] * n_am1_layers + ["Identity"],
                    ),
                    postprocess_layer=ComputePartialCharges(),
                )
            },
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

    def compute_charges(self, molecule: Molecule) -> torch.Tensor:

        atom_features, bond_features = self.features()
        dgl_molecule = DGLMolecule.from_openff(molecule, atom_features, bond_features)

        return self.forward(dgl_molecule)[self._charge_readout]
