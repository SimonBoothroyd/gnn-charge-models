import functools
import json

import click
import numpy
import pandas
import rich
import seaborn
from matplotlib import pyplot
from nagl.utilities.toolkits import capture_toolkit_warnings
from openff.toolkit.topology import Molecule
from openff.units import unit
from rich import pretty


@functools.lru_cache
def has_net_charge(smiles: str) -> bool:

    from openmm import unit as openmm_unit

    with capture_toolkit_warnings():
        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        return not numpy.isclose(
            molecule.total_charge.value_in_unit(openmm_unit.elementary_charge),
            0.0,
            atol=1.0e-3,
        )


@click.command()
@click.option(
    "--input",
    "input_tuples",
    type=(str, click.Path(exists=True, file_okay=True, dir_okay=False)),
    multiple=True,
)
@click.option(
    "--reference",
    "reference_tuple",
    type=(str, click.Path(exists=True, file_okay=True, dir_okay=False)),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=False,
)
@click.option(
    "--show/--no-show",
    "show_plot",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--neutral/--all",
    "neutral_only",
    type=bool,
    default=False,
    show_default=True,
)
def main(input_tuples, reference_tuple, output_path, show_plot, neutral_only):

    console = rich.get_console()
    pretty.install(console)

    plot_data_rows = []
    reference_data, reference_label = None, None

    if reference_tuple is not None:
        reference_label, reference_path = reference_tuple

        with open(reference_path) as file:
            reference_data = json.load(file)

    x_label = (
        "RMSE (kcal / mol)"
        if reference_data is None
        else f"RMSE - RMSE$_{{{reference_label}}}$ (kcal / mol)"
    )

    n_charged = 0

    for label, input_path in input_tuples:

        with open(input_path) as file:
            per_molecule_rmse = json.load(file)

        if reference_data is not None:

            for smiles in per_molecule_rmse:
                per_molecule_rmse[smiles] -= reference_data[smiles]

        for smiles, rmse in per_molecule_rmse.items():

            if neutral_only and has_net_charge(smiles):
                n_charged += 1
                continue

            conversion = (1.0 * unit.avogadro_constant * unit.hartree).m_as(
                unit.kilocalorie / unit.mole
            )

            plot_data_rows.append(
                {
                    x_label: rmse * conversion,
                    "SMILES": smiles,
                    "label": label,
                }
            )

    if neutral_only:
        console.print(f"{n_charged} molecules had a net charge and were skipped")

    plot_data = pandas.DataFrame(plot_data_rows)

    seaborn.histplot(plot_data, x=x_label, hue="label", element="step")

    if output_path is not None:
        pyplot.tight_layout()
        pyplot.savefig(output_path)
    if show_plot:
        pyplot.show()


if __name__ == "__main__":
    main()
