import json
import os
from pathlib import Path
from typing import List, Tuple

import click
import rich

from absolv.models import EquilibriumProtocol, State, System, TransferFreeEnergySchema
from absolv.runners.equilibrium import EquilibriumRunner
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit
from openff.utilities import temporary_cd
from rich import pretty


@click.command()
@click.option(
    "--input-systems",
    "input_systems_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--input-index",
    type=int
)
@click.option(
    "--force-field",
    "force_field_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False)
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False)
)
@click.option(
    "--tmp-dir",
    "working_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True)
)
def main(
    input_systems_path, input_index, force_field_path, output_path, working_directory
):

    console = rich.get_console()
    pretty.install(console)

    force_field = ForceField(force_field_path, allow_cosmetic_attributes=True)

    with open(input_systems_path, "r") as file:
        input_systems: List[Tuple[str, str]] = json.load(file)

    solute_smiles, solvent_smiles = input_systems[input_index]

    schema = TransferFreeEnergySchema(
        system=System(
            solutes={solute_smiles: 1}, solvent_a=None, solvent_b={solvent_smiles: 1000}
        ),
        state=State(temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere),
        alchemical_protocol_a=EquilibriumProtocol(
            lambda_sterics=[1.0, 1.0, 1.0, 1.0, 1.0],
            lambda_electrostatics=[1.0, 0.75, 0.5, 0.25, 0.0],
            sampler="repex"
        ),
        alchemical_protocol_b=EquilibriumProtocol(
            lambda_sterics=[
                1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40,
                0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
            ],
            lambda_electrostatics=[
                1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            ],
            sampler="repex"
        ),
    )

    if working_directory is not None:
        Path(working_directory).mkdir(parents=True, exist_ok=True)

    with temporary_cd(working_directory):
        console.print(f"running in {os.getcwd()}")

        EquilibriumRunner.setup(schema, force_field)
        EquilibriumRunner.run(platform="CUDA")

        result = EquilibriumRunner.analyze()

    output_path = Path(output_path)
    console.print(f"saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        file.write(result.json(indent=2))


if __name__ == '__main__':
    main()
