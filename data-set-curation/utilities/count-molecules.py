import click
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def main(input_path):

    with capture_toolkit_warnings():
        n_molecules = sum(1 for _ in stream_from_file(input_path, as_smiles=True))

    print(n_molecules)


if __name__ == "__main__":
    main()
