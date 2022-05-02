import functools
import json
import os.path
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple
from urllib.request import urlopen

import click
import numpy
import rich
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.bespokefit.optimizers.forcebalance import ForceBalanceInputFactory
from openff.bespokefit.schema.fitting import OptimizationStageSchema
from openff.bespokefit.schema.optimizers import ForceBalanceSchema
from openff.bespokefit.schema.smirnoff import (
    AngleHyperparameters,
    AngleSMIRKS,
    BondHyperparameters,
    BondSMIRKS,
    ProperTorsionHyperparameters,
    ProperTorsionSMIRKS,
)
from openff.bespokefit.schema.targets import (
    OptGeoTargetSchema,
    TorsionProfileTargetSchema,
)
from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils import UndefinedStereochemistryError
from openmm import unit
from rich import pretty
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    track,
)


def download_file(url: str, description: str) -> bytes:
    """Downloads a file while showing a pretty ``rich`` progress bar."""

    with Progress(
        TextColumn(f"[bold blue]{description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=False,
    ) as progress:

        task_id = progress.add_task("download", start=False)

        with urlopen(url) as response:

            progress.update(task_id, total=int(response.info()["Content-length"]))

            with BytesIO() as buffer:

                progress.start_task(task_id)

                for data in iter(functools.partial(response.read, 32768), b""):

                    buffer.write(data)
                    progress.update(task_id, advance=len(data))

                return buffer.getvalue()


def can_assign_charges(cmiles: str) -> Tuple[str, bool]:

    with capture_toolkit_warnings():

        try:
            molecule: Molecule = Molecule.from_smiles(
                cmiles, allow_undefined_stereo=True
            )
            molecule.generate_conformers(n_conformers=1)
            molecule.assign_partial_charges(
                "am1bcc", use_conformers=molecule.conformers
            )
        except:
            return cmiles, False

        return cmiles, True


@click.command()
@click.option(
    "--input-ff",
    "force_field_path",
    help="The path to the initial force field",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-name",
    "output_name",
    help="The name of the optimize to save.",
    type=str,
    required=True,
)
@click.option(
    "--output-dir",
    "output_directory",
    help="The directory to save the inputs in.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="",
    show_default=True,
)
def main(force_field_path, output_name, output_directory):

    console = rich.get_console()
    pretty.install(console)

    torsion_training_set = TorsionDriveResultCollection.parse_raw(
        download_file(
            "https://raw.githubusercontent.com/openforcefield/openff-sage/"
            "325460b254c3532b96ae43deb5a3a963605609b6/"
            "data-set-curation/quantum-chemical/data-sets/1-2-0-td-set.json",
            "downloading TD train set",
        )
    )
    torsion_id_to_cmiles = {
        entry.record_id: entry.cmiles
        for entries in torsion_training_set.entries.values()
        for entry in entries
    }

    optimization_training_set = OptimizationResultCollection.parse_raw(
        download_file(
            "https://raw.githubusercontent.com/openforcefield/openff-sage/"
            "325460b254c3532b96ae43deb5a3a963605609b6/"
            "data-set-curation/quantum-chemical/data-sets/1-2-0-opt-set-v3.json",
            "downloading OPT train set",
        )
    )

    unique_smiles = sorted(
        {
            entry.cmiles
            for training_set in [torsion_training_set, optimization_training_set]
            for entries in training_set.entries.values()
            for entry in entries
        }
    )

    with Pool(processes=10) as pool:

        cmiles_to_exclude = [
            cmiles
            for cmiles, can_charge in track(
                pool.imap(can_assign_charges, unique_smiles),
                total=len(unique_smiles),
                description="checking can charge",
            )
            if not can_charge
            # the OpenFF recharge BCCs don't cover P yet.
            or "P" in cmiles
        ]

    console.print(
        f"{len(cmiles_to_exclude)} molecules could not be assigned charges by OE and "
        f"so were removed:"
    )
    console.print(cmiles_to_exclude)

    with console.status("filtering out un-chargable SMILES"):

        n_original = sum(
            len(entries) for entries in torsion_training_set.entries.values()
        )
        torsion_training_set.entries = {
            address: [
                entry for entry in entries if entry.cmiles not in cmiles_to_exclude
            ]
            for address, entries in torsion_training_set.entries.items()
        }
        n_after = sum(len(entries) for entries in torsion_training_set.entries.values())
        console.print(f"{n_original - n_after} torsion drive records were filtered")

        n_original = sum(
            len(entries) for entries in optimization_training_set.entries.values()
        )
        optimization_training_set.entries = {
            address: [
                entry for entry in entries if entry.cmiles not in cmiles_to_exclude
            ]
            for address, entries in optimization_training_set.entries.items()
        }
        n_after = sum(
            len(entries) for entries in optimization_training_set.entries.values()
        )
        console.print(f"{n_original - n_after} optimization records were filtered")

    initial_force_field = ForceField(force_field_path, allow_cosmetic_attributes=True)
    initial_force_field.deregister_parameter_handler("Constraints")

    # Define the parameters to train
    valence_smirks = json.loads(
        download_file(
            "https://raw.githubusercontent.com/openforcefield/openff-sage/"
            "325460b254c3532b96ae43deb5a3a963605609b6/"
            "data-set-curation/quantum-chemical/data-sets/"
            "1-2-0-opt-set-v2-valence-smirks.json",
            "downloading valence SMIRKS",
        )
    )
    torsion_smirks = json.loads(
        download_file(
            "https://raw.githubusercontent.com/openforcefield/openff-sage/"
            "325460b254c3532b96ae43deb5a3a963605609b6/"
            "data-set-curation/quantum-chemical/data-sets/"
            "1-2-0-td-set-torsion-smirks.json",
            "downloading torsion SMIRKS",
        )
    )

    parameters_to_train = [
        *[
            BondSMIRKS(smirks=smirks, attributes={"k", "length"})
            for smirks in valence_smirks["Bonds"]
        ],
        *[
            (
                AngleSMIRKS(smirks=smirks, attributes={"k", "angle"})
                if not numpy.isclose(
                    # See https://github.com/leeping/forcebalance/issues/258
                    initial_force_field["Angles"]
                    .parameters[smirks]
                    .angle.value_in_unit(unit.degrees),
                    180.0,
                )
                else AngleSMIRKS(smirks=smirks, attributes={"k"})
            )
            for smirks in valence_smirks["Angles"]
        ],
        *[
            ProperTorsionSMIRKS(
                smirks=smirks,
                attributes={
                    f"k{i + 1}"
                    for i in range(
                        len(
                            initial_force_field.get_parameter_handler("ProperTorsions")
                            .parameters[smirks]
                            .k
                        )
                    )
                },
            )
            for smirks in torsion_smirks["ProperTorsions"]
        ],
    ]

    # Define the full schema for the optimization.
    optimization_schema = OptimizationStageSchema(
        optimizer=ForceBalanceSchema(
            max_iterations=50,
            step_convergence_threshold=0.01,
            objective_convergence_threshold=0.1,
            gradient_convergence_threshold=0.1,
            n_criteria=2,
            initial_trust_radius=-1.0,
            extras={"wq_port": "55125", "asynchronous": "True"},
            penalty_type="L2",
        ),
        parameters=parameters_to_train,
        parameter_hyperparameters=[
            BondHyperparameters(priors={"k": 1.0e02, "length": 1.0e-01}),
            # See openforcefield/openff-bespokefit/pull/169
            AngleHyperparameters.construct(priors={"k": 1.0e02, "angle": 2.0e01}),
            ProperTorsionHyperparameters(priors={"k": 1.0}),
        ],
        targets=[
            TorsionProfileTargetSchema(
                reference_data=torsion_training_set,
                energy_denominator=1.0,
                energy_cutoff=5.0,
                extras={"remote": "1"},
            ),
            OptGeoTargetSchema(
                reference_data=optimization_training_set,
                weight=0.1,
                extras={"batch_size": 30, "remote": "1"},
            ),
        ],
    )

    with console.status("building inputs"):

        ForceBalanceInputFactory.generate(
            os.path.join(output_directory, output_name),
            optimization_schema,
            ForceField(initial_force_field.to_string(discard_cosmetic_attributes=True)),
        )

    root_directory = Path(output_directory, output_name)
    molecule_files = [*root_directory.glob("**/*.sdf")]

    for molecule_file in track(molecule_files, description="fixing OE stereo"):

        molecule_file = str(molecule_file)

        try:
            Molecule.from_file(str(molecule_file), "SDF")
        except UndefinedStereochemistryError:

            console.print(f"fixing {molecule_file}")

            if "torsion-" not in molecule_file:
                raise NotImplementedError()

            record_id = molecule_file.split("-")[-1].split("/")[0]

            old_molecule = Molecule.from_file(
                str(molecule_file), "SDF", allow_undefined_stereo=True
            )
            new_molecule: Molecule = Molecule.from_smiles(
                torsion_id_to_cmiles[record_id]
            )

            _, mapping = Molecule.are_isomorphic(
                new_molecule,
                old_molecule,
                return_atom_map=True,
                bond_stereochemistry_matching=False,
            )
            new_molecule = new_molecule.remap(mapping)
            new_molecule._properties = old_molecule._properties

            new_molecule.to_file(molecule_file, "SDF")


if __name__ == "__main__":
    main()
