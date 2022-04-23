import json

import click
import pandas
import seaborn
from matplotlib import pyplot
from openff.units import unit


@click.command()
@click.option(
    "--input",
    "input_paths",
    type=(str, click.Path(exists=True, file_okay=True, dir_okay=False)),
    multiple=True,
)
@click.option(
    "--output",
    "output_path",
    type=(str, click.Path(exists=False, file_okay=True, dir_okay=False)),
    required=False,
)
@click.option(
    "--show/--no-show",
    "show_plot",
    type=bool,
    default=True,
    show_default=True,
)
def main(input_paths, output_path, show_plot):

    plot_data_rows = []

    with open("no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json") as file:
        reference = json.load(file)

    for label, input_path in input_paths:

        with open(input_path) as file:
            per_molecule_rmse = json.load(file)

        for s in per_molecule_rmse:
            per_molecule_rmse[s] -= reference[s]

        for smiles, rmse in per_molecule_rmse.items():

            conversion = (1.0 * unit.avogadro_constant * unit.hartree).m_as(
                unit.kilocalorie / unit.mole
            )

            plot_data_rows.append(
                {
                    "RMSE (kcal / mol)": rmse * conversion,
                    "SMILES": smiles,
                    "label": label,
                }
            )

    plot_data = pandas.DataFrame(plot_data_rows)

    seaborn.histplot(plot_data, x="RMSE (kcal / mol)", hue="label", element="step")

    if output_path is not None:
        pyplot.tight_layout()
        pyplot.savefig(output_path)
    if show_plot:
        pyplot.show()


if __name__ == "__main__":
    main()
