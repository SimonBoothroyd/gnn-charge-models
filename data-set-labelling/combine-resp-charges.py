import os.path
import pickle
from glob import glob

import numpy
from nagl.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
)
from openff.toolkit.topology import Molecule


def main():

    records_by_inchi_key = {}

    for charge_path in glob(
        os.path.join("resp-charges", "fragment-set", "esp-fragment-records-*.pkl")
    ):

        with open(charge_path, "rb") as file:
            charge_set = pickle.load(file)

        for smiles, charges in charge_set:

            molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            inchi_key = molecule.to_inchikey(fixed_hydrogens=True)

            records_by_inchi_key[inchi_key] = MoleculeRecord(
                smiles=smiles,
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.zeros((len(charges), 3)),
                        partial_charges=[
                            PartialChargeSet(
                                method="resp", values=charges.flatten().tolist()
                            )
                        ],
                    )
                ],
            )

    record_store = MoleculeStore(os.path.join("resp-charges", "fragment-set.sqlite"))
    record_store.store(*records_by_inchi_key.values())


if __name__ == "__main__":
    main()
