import os.path
import pickle
from pprint import pprint

import click
import pytorch_lightning as pl
import rich
import torch
from click_option_group import optgroup
from nagl.features import (
    AtomConnectivity,
    AtomFeature,
    AtomicElement,
    AtomIsInRing,
    BondIsInRing,
)
from nagl.lightning import DGLMoleculeDataModule, DGLMoleculeLightningModel
from nagl.nn import SequentialLayers
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.postprocess import ComputePartialCharges
from nagl.resonance import enumerate_resonance_forms
from nagl.utilities.toolkits import normalize_molecule
from openff.toolkit.topology import Molecule
from pytorch_lightning.loggers import TensorBoardLogger
from rich import pretty
from rich.console import NewLine


class AtomAverageFormalCharge(AtomFeature):
    """Computes the average formal charge on each atom in a molecule across resonance
    structures."""

    def __call__(self, molecule: Molecule) -> torch.Tensor:

        try:
            molecule = normalize_molecule(molecule)
        except AssertionError:
            # See openff-toolkit/issues/1181
            pass

        resonance_forms = enumerate_resonance_forms(
            molecule,
            as_dicts=True,
            # exclude for e.g. the charged resonance form of an amide
            lowest_energy_only=True,
            # exclude resonance structures that only differ in things like kekule
            # form
            include_all_transfer_pathways=False,
        )

        formal_charges = [
            [
                atom["formal_charge"]
                for resonance_form in resonance_forms
                if i in resonance_form["atoms"]
                for atom in resonance_form["atoms"][i]
            ]
            for i in range(molecule.n_atoms)
        ]

        feature_tensor = torch.tensor(
            [
                [
                    sum(formal_charges[i]) / len(formal_charges[i])
                    if len(formal_charges[i]) > 0
                    else 0.0
                ]
                for i in range(molecule.n_atoms)
            ]
        )

        return feature_tensor

    def __len__(self):
        return 1


@optgroup.group("Training set")
@optgroup.option(
    "--train-set",
    "train_set_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    multiple=True,
)
@optgroup.option(
    "--train-batch-size",
    type=click.INT,
    default=512,
    show_default=True,
    required=True,
)
@optgroup.group("Validation set")
@optgroup.option(
    "--val-set",
    "val_set_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    multiple=True,
)
@optgroup.group("Test set")
@optgroup.option(
    "--test-set",
    "test_set_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    multiple=True,
)
@optgroup.option(
    "--charge-method",
    "partial_charge_method",
    type=click.STRING,
    required=True,
)
@optgroup.group("Model")
@optgroup.option(
    "--n-gcn-layers",
    type=click.INT,
    default=4,
    show_default=True,
    required=True,
)
@optgroup.option(
    "--n-gcn-hidden-features",
    type=click.INT,
    default=64,
    show_default=True,
    required=True,
)
@optgroup.option(
    "--n-am1-layers",
    type=click.INT,
    default=4,
    show_default=True,
    required=True,
)
@optgroup.option(
    "--n-am1-hidden-features",
    type=click.INT,
    default=64,
    show_default=True,
    required=True,
)
@optgroup.group("Optimizer")
@optgroup.option(
    "--n-epochs",
    type=click.INT,
    default=500,
    show_default=True,
    required=True,
)
@optgroup.option(
    "--learning-rate",
    type=click.FLOAT,
    default=1.0e-4,
    show_default=True,
    required=True,
)
@optgroup.group("Other")
@optgroup.option(
    "--seed",
    type=click.INT,
    required=False,
)
@optgroup.option(
    "--output-dir",
    "output_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="lightning-logs",
    show_default=True,
    required=False,
)
@click.command()
def main(
    train_set_path,
    train_batch_size,
    val_set_path,
    test_set_path,
    partial_charge_method,
    n_gcn_layers,
    n_gcn_hidden_features,
    n_am1_layers,
    n_am1_hidden_features,
    learning_rate,
    n_epochs,
    seed,
    output_directory,
):

    cli_inputs = locals()

    console = rich.get_console()
    pretty.install(console)

    console.rule("CLI inputs")
    console.print(NewLine())
    pprint(cli_inputs)
    console.print(NewLine())

    train_set_path = None if len(train_set_path) == 0 else train_set_path
    val_set_path = None if len(val_set_path) == 0 else val_set_path
    test_set_path = None if len(test_set_path) == 0 else test_set_path

    if seed is not None:
        pl.seed_everything(seed)

    # Define the features of interest.
    atom_features = [
        AtomicElement(["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]),
        AtomConnectivity(),
        AtomAverageFormalCharge(),
        AtomIsInRing(),
    ]
    bond_features = [BondIsInRing()]

    # Load in the pre-processed training and test molecules and store them in
    # featurized graphs.
    data_module = DGLMoleculeDataModule(
        atom_features,
        bond_features,
        partial_charge_method=partial_charge_method,
        bond_order_method=None,
        train_set_path=train_set_path,
        train_batch_size=train_batch_size,
        val_set_path=val_set_path,
        val_batch_size=None,
        test_set_path=test_set_path,
        test_batch_size=None,
        use_cached_data=True,
    )
    n_atom_features = data_module.n_atom_features

    # Define the model.
    model = DGLMoleculeLightningModel(
        convolution_module=ConvolutionModule(
            architecture="SAGEConv",
            in_feats=n_atom_features,
            hidden_feats=[n_gcn_hidden_features] * n_gcn_layers,
        ),
        readout_modules={
            f"{partial_charge_method}-charges": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(
                    in_feats=n_gcn_hidden_features,
                    hidden_feats=[n_am1_hidden_features] * n_am1_layers + [2],
                    activation=["ReLU"] * n_am1_layers + ["Identity"],
                ),
                postprocess_layer=ComputePartialCharges(),
            )
        },
        learning_rate=learning_rate,
    )

    console.print(NewLine())
    console.rule("model")
    console.print(NewLine())
    console.print(model)

    console.print(NewLine())
    console.rule("training")
    console.print(NewLine())

    # Train the model
    n_gpus = 0 if not torch.cuda.is_available() else 1
    print(f"Using {n_gpus} GPUs")

    version_string = (
        f"{train_batch_size}-"
        f"{n_gcn_layers}-"
        f"{n_gcn_hidden_features}-"
        f"{n_am1_layers}-"
        f"{n_am1_hidden_features}-"
        f"{learning_rate}"
    )

    os.makedirs(output_directory, exist_ok=True)
    logger = TensorBoardLogger(output_directory, version=version_string)

    trainer = pl.Trainer(
        gpus=n_gpus, min_epochs=n_epochs, max_epochs=n_epochs, logger=logger
    )

    trainer.fit(model, datamodule=data_module)

    with open(
        os.path.join(output_directory, "default", version_string, "metrics.pkl"), "wb"
    ) as file:

        pickle.dump((trainer.callback_metrics, trainer.logged_metrics), file)

    if test_set_path is not None:
        trainer.test(model, data_module)


if __name__ == "__main__":
    main()