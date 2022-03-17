import json
import os.path
import pickle
import traceback
from typing import List, Set, Tuple

import click
import mdtraj
import numpy
from nagl.utilities.toolkits import stream_from_file
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import (
    OpenEyeToolkitWrapper,
    ToolkitRegistry,
    UndefinedStereochemistryError,
)
from openff.utilities import temporary_cd
from qcelemental.models.common_models import DriverEnum, Model
from qcelemental.models.molecule import Molecule as QCMolecule
from qcelemental.models.procedures import (
    OptimizationInput,
    OptimizationResult,
    QCInputSpecification,
)
from qcelemental.molutil import guess_connectivity
from qcengine import compute_procedure
from simtk import unit
from tqdm import tqdm


def _find_h_bonds(molecule: Molecule, conformer: unit.Quantity) -> Set[Tuple[int, int]]:

    conformer = conformer.value_in_unit(unit.nanometers).tolist()

    mdtraj_topology = mdtraj.Topology.from_openmm(molecule.to_topology().to_openmm())
    mdtraj_trajectory = mdtraj.Trajectory(
        numpy.array([conformer]) * unit.nanometers, mdtraj_topology
    )

    h_bonds = mdtraj.baker_hubbard(mdtraj_trajectory, freq=0.0, periodic=False)

    return {(h_index, a_index) for _, h_index, a_index in h_bonds}


def _validate_optimization(
    molecule: Molecule,
    initial_connectivity: Set[Tuple[int, int]],
    initial_h_bonds: Set[Tuple[int, int]],
    final_molecule: QCMolecule,
) -> bool:

    smiles = molecule.to_smiles(explicit_hydrogens=False)

    final_connectivity = {
        tuple(sorted(connection))
        for connection in guess_connectivity(
            final_molecule.symbols, final_molecule.geometry
        )
    }
    if initial_connectivity != final_connectivity:

        print(
            f"connectivity of {smiles} changed - "
            f"old={initial_connectivity} new={final_connectivity}"
        )
        return False

    final_geometry = final_molecule.geometry.reshape(-1, 3) * unit.bohr
    final_h_bonds = _find_h_bonds(molecule, final_geometry)

    if initial_h_bonds != final_h_bonds:

        print(f"h-bonding has changed - old={initial_h_bonds} new={final_h_bonds}")
        return False

    return True


def _generate_conformers(
    smiles: str,
    n_conformers: int,
    qc_settings: List[Tuple[str, str, str]],
    n_processes: int,
    memory: int,
) -> unit.Quantity:

    toolkit_registry = ToolkitRegistry([OpenEyeToolkitWrapper()])

    # 1. Generate a diverse set of ELF conformers.
    try:
        off_molecule: Molecule = Molecule.from_smiles(smiles)
    except UndefinedStereochemistryError:

        off_molecule: Molecule = Molecule.from_smiles(
            smiles, allow_undefined_stereo=True
        )
        stereoisomers = off_molecule.enumerate_stereoisomers(
            undefined_only=True, max_isomers=1
        )

        if len(stereoisomers) > 0:
            off_molecule = stereoisomers[0]

    off_molecule.generate_conformers(
        n_conformers=500,
        rms_cutoff=0.5 * unit.angstrom,
        toolkit_registry=toolkit_registry,
    )
    off_molecule.apply_elf_conformer_selection(toolkit_registry=toolkit_registry)

    initial_connectivity = {
        tuple(sorted([bond.atom1_index, bond.atom2_index]))
        for bond in off_molecule.bonds
    }

    final_conformers = []

    for i, conformer in enumerate(
        tqdm(off_molecule.conformers[:n_conformers], desc="CONF")
    ):
        initial_h_bonds = _find_h_bonds(off_molecule, conformer)
        initial_molecule = off_molecule.to_qcschema(conformer=i)

        current_molecule = initial_molecule

        # 2. minimize the conformer using the requested QC settings.
        for program, method, basis in qc_settings:

            optimization_input = OptimizationInput(
                keywords={
                    "program": program,
                    "coordsys": "dlc",
                    "convergence_set": "GAU_LOOSE",
                    "maxiter": 300,
                },
                input_specification=QCInputSpecification(
                    model=Model(method=method, basis=basis),
                    driver=DriverEnum.gradient,
                ),
                initial_molecule=current_molecule,
            )
            # noinspection PyTypeChecker
            result: OptimizationResult = compute_procedure(
                optimization_input,
                "geometric",
                raise_error=True,
                local_options={
                    "ncores": n_processes,
                    "nnodes": 1,
                    "jobs_per_node": 1,
                    "memory": memory,
                },
            )

            final_molecule: QCMolecule = result.trajectory[-1].molecule
            current_molecule = final_molecule

        is_valid = _validate_optimization(
            off_molecule, initial_connectivity, initial_h_bonds, final_molecule
        )

        if not is_valid:
            tqdm.write(f"conformer {i} did not validate")
            continue

        final_conformers.append(final_molecule.geometry.reshape(-1, 3))

    return final_conformers * unit.bohr


@click.command()
@click.option(
    "--input",
    "input_path",
    help="The path to the input molecules. This should either be an SDF, a GZipped "
    "SDF, or a SMI file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_directory",
    help="The directory to store the output in.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--n-conformers",
    type=int,
    default=5,
    help="The maximum number of ELF conformers to generate per molecule.",
    show_default=True,
)
@click.option(
    "--qc-settings",
    help="Settings that describe the program, method, and basis to use when minimizing "
    "each conformer.",
    type=(str, str, str),
    default=("psi4", "hf", "6-31G*"),
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    help="The size of the batch to compute.",
    show_default=True,
)
@click.option(
    "--batch-idx",
    "batch_index",
    type=int,
    default=0,
    help="The (zero-based) index of the batch to compute.",
    show_default=True,
)
@click.option(
    "--n-processes",
    type=int,
    default=16,
    help="The number of processes to parallelize psi4 across.",
    show_default=True,
)
@click.option(
    "--memory",
    type=int,
    default=256,
    help="The maximum memery available to psi4 in GiB.",
    show_default=True,
)
def main(
    input_path: str,
    output_directory: str,
    n_conformers: int,
    qc_settings: Tuple[str, str, str],
    batch_size: int,
    batch_index: int,
    n_processes: int,
    memory: int,
):

    smiles_list = [*stream_from_file(input_path, as_smiles=True)]
    smiles_list = smiles_list[batch_index * batch_size : (batch_index + 1) * batch_size]

    qc_settings = [qc_settings]

    completed, failed = [], []

    os.makedirs(output_directory, exist_ok=True)

    with temporary_cd(output_directory):

        if os.path.isfile(f"completed-{batch_index}.pkl"):

            with open(f"completed-{batch_index}.pkl", "rb") as file:
                completed = pickle.load(file)

            tqdm.write(f"{len(completed)} SMILES already complete")

            for smiles, _ in completed:
                assert smiles in smiles_list
                smiles_list.remove(smiles)

        for smiles in tqdm(smiles_list, desc="SMILES"):

            try:
                conformers = _generate_conformers(
                    smiles, n_conformers, qc_settings, n_processes, memory
                )
                completed.append((smiles, conformers))
            except BaseException as e:

                failed.append(
                    (
                        smiles,
                        "\n".join(
                            traceback.format_exception(type(e), e, e.__traceback__)
                        ),
                    )
                )

            with open(f"completed-{batch_index}.pkl", "wb") as file:
                pickle.dump(completed, file)

            with open(f"failed-{batch_index}.json", "w") as file:
                json.dump(failed, file)


if __name__ == "__main__":
    main()
