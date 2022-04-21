import json
from pathlib import Path
from typing import Tuple, Union

from openff.recharge.aromaticity import AromaticityModels
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCParameter,
    original_am1bcc_corrections,
)
from openff.recharge.charges.library import LibraryChargeCollection
from openff.recharge.charges.qc import QCChargeSettings
from openff.recharge.charges.vsite import (
    BondChargeSiteParameter,
    DivalentLonePairParameter,
    VirtualSiteCollection,
)
from openff.recharge.conformers import ConformerSettings

ChargeCollection = Union[
    Tuple[ConformerSettings, QCChargeSettings], LibraryChargeCollection
]


def save_parameters(
    output_directory: Path,
    charge_collection: ChargeCollection,
    bcc_collection: BCCCollection,
    v_site_collection: VirtualSiteCollection,
):

    output_directory.mkdir(parents=True, exist_ok=True)

    with Path(output_directory, "initial-parameters-base.json").open("w") as file:

        if isinstance(charge_collection, LibraryChargeCollection):
            file.write(charge_collection.json(indent=2))
        else:
            json.dump(
                (charge_collection[0].dict(), charge_collection[1].dict()),
                file,
                indent=2,
            )

    with Path(output_directory, "initial-parameters-bcc.json").open("w") as file:
        file.write(bcc_collection.json(indent=2))
    with Path(output_directory, "initial-parameters-v-site.json").open("w") as file:
        file.write(v_site_collection.json(indent=2))


def main():

    conformer_settings = ConformerSettings(
        method="omega-elf10", sampling_mode="dense", max_conformers=10
    )

    # ============ Virtual sites only on pyridine ============

    save_parameters(
        output_directory=Path("sp2-pyridine-only", "no-v-sites"),
        charge_collection=(conformer_settings, QCChargeSettings(theory="am1")),
        bcc_collection=original_am1bcc_corrections(),
        v_site_collection=VirtualSiteCollection(parameters=[]),
    )
    save_parameters(
        output_directory=Path("sp2-pyridine-only", "v-sites"),
        charge_collection=(conformer_settings, QCChargeSettings(theory="am1")),
        bcc_collection=BCCCollection(
            parameters=[
                BCCParameter(
                    smirks=(
                        "[#6X3H1a:1]1:"
                        "[#7X2a:2]:"
                        "[#6X3H1a]:"
                        "[#6X3a]:"
                        "[#6X3a]:"
                        "[#6X3a]1"
                    ),
                    value=0.0,
                    provenance=None,
                ),
                BCCParameter(
                    smirks=(
                        "[#1:2]"
                        "[#6X3H1a:1]1:"
                        "[#7X2a]:"
                        "[#6X3H1a]:"
                        "[#6X3a]:"
                        "[#6X3a]:"
                        "[#6X3a]1"
                    ),
                    value=0.0,
                    provenance=None,
                ),
                *original_am1bcc_corrections().parameters,
            ],
            aromaticity_model=AromaticityModels.AM1BCC,
        ),
        v_site_collection=VirtualSiteCollection(
            parameters=[
                DivalentLonePairParameter(
                    smirks=(
                        "[#6X3H1a:2]1:"
                        "[#7X2a:1]:"
                        "[#6X3H1a:3]:"
                        "[#6X3a]:"
                        "[#6X3a]:"
                        "[#6X3a]1"
                    ),
                    name="EP",
                    distance=0.35,
                    out_of_plane_angle=0.0,
                    charge_increments=(0.0, 0.35, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="once",
                )
            ]
        ),
    )

    # ============ Halogens ============

    save_parameters(
        output_directory=Path("halogens", "no-v-sites"),
        charge_collection=(conformer_settings, QCChargeSettings(theory="am1")),
        bcc_collection=original_am1bcc_corrections(),
        v_site_collection=VirtualSiteCollection(parameters=[]),
    )
    save_parameters(
        output_directory=Path("halogens", "v-sites"),
        charge_collection=(conformer_settings, QCChargeSettings(theory="am1")),
        bcc_collection=original_am1bcc_corrections(),
        v_site_collection=VirtualSiteCollection(
            parameters=[
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#17:1]",
                    name="EP1",
                    distance=0.35,
                    charge_increments=(0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#17:1]",
                    name="EP2",
                    distance=1.0,
                    charge_increments=(-0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#35:1]",
                    name="EP1",
                    distance=0.35,
                    charge_increments=(0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
                BondChargeSiteParameter(
                    smirks="[#6A:2]-[#35:1]",
                    name="EP2",
                    distance=1.0,
                    charge_increments=(-0.05, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                ),
            ]
        ),
    )


if __name__ == "__main__":
    main()
