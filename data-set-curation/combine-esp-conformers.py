import json
import os.path
import pickle
from collections import defaultdict
from glob import glob

import click


@click.option(
    "--input",
    "input_directory",
    help="The path to the directory containing the completed conformers and " "errors.",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_directory",
    help="The path to the directory to save the gathered conformers / errors in.",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.command()
def main(input_directory, output_directory):

    output_name = os.path.basename(input_directory)

    errors_by_type = defaultdict(list)

    for error_file in glob(os.path.join(input_directory, "failed-*.json")):

        with open(error_file) as file:
            errors = json.load(file)

        for error in errors:

            message: str
            _, message = error

            if "Shell Entries: ['I']" in message:
                errors_by_type["iodine"].append(error)
            elif "OpenEye Omega conformer generation failed" in message:
                errors_by_type["omega"].append(error)
            elif message.find("Unable to make OFFMol from OEMol") != message.rfind(
                "Unable to make OFFMol from OEMol"
            ):
                errors_by_type["stereo"].append(error)
            else:
                errors_by_type["unexpected"].append(error)

    print("N ERRORS: ", sum(len(errors) for errors in errors_by_type.values()))

    with open(
        os.path.join(output_directory, f"{output_name}-errors.json"), "w"
    ) as file:
        json.dump(errors_by_type, file)

    completed = []

    for completed_file in glob(os.path.join(input_directory, "completed-*.pkl")):

        with open(completed_file, "rb") as file:
            completed.extend(pickle.load(file))

    print("N MOLECULES COMPLETE: ", len(completed))
    print("N CONFORMERS: ", sum(len(conformers) for _, conformers in completed))

    with open(
        os.path.join(output_directory, f"{output_name}-complete.smi"), "w"
    ) as file:
        file.write("\n".join(smiles for smiles, _ in completed))

    with open(
        os.path.join(output_directory, f"{output_name}-complete.pkl"), "wb"
    ) as file:
        pickle.dump(completed, file)


if __name__ == "__main__":
    main()
