import click
from openff.recharge.grids import LatticeGridSettings, MSKGridSettings


@click.command()
def generate():

    grid_settings = {
        "msk-default.json": MSKGridSettings(),
        "fcc-default.json": LatticeGridSettings(),
    }

    for file_name, settings in grid_settings.items():

        with open(file_name, "w") as file:
            file.write(settings.json(indent=4))


if __name__ == "__main__":
    generate()
