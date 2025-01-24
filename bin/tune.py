import sys
sys.path.append(".")
import gc
import torch
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import click
import itertools
import wandb
from lightning.pytorch.tuner import Tuner

from config import from_toml
from model import get_model
from data.datamodule import PFresGODataModule

def _tune(config, subontology, wandb_run_name, go_release):
    config_file_name = Path(config).name.split(".")[0]
    model_name = config_file_name.replace("_", "-")
    config = from_toml(config)

    if go_release is not None:
        print("Overriding GO release to", go_release)
        config.train.go_release = go_release

    if subontology is not None:
        if subontology in ["bp", "cc", "mf"]:
            subontology = "biological_process" if subontology == "bp" else "cellular_component" if subontology == "cc" else "molecular_function"

        if subontology not in ["biological_process", "cellular_component", "molecular_function"]:
            raise ValueError(f"Invalid subontology override: {subontology}")
        print("Overriding subontology to", subontology)
        config.train.subontology = subontology

    subontology_short = "".join([word[0] for word in config.train.subontology.split("_")])

    if wandb_run_name is not None:
        print("Overriding wandb run name to", wandb_run_name)
        wandb_run_name = wandb_run_name
    else:
        wandb_run_name = f"tune-contempro-{subontology_short}-{config.train.go_release}-{model_name}"

    logger = WandbLogger(project="contempro", name=wandb_run_name)

    dm = PFresGODataModule(
        data_dir=config.train.data_dir,
        batch_size=config.train.batch_size,
        num_workers=config.train.dm_num_workers,
        ontology=config.train.subontology,
        go_release=config.train.go_release,
        order_go_terms=config.train.order_go_terms,
    )

    model = get_model(config.model.name, config)

    trainer = Trainer(
        max_epochs=5,
        logger=[logger],
        accelerator="auto"
    )

    tuner = Tuner(trainer)

    # Find optimal learning rate
    lr_finder = tuner.lr_find(model, datamodule=dm)
    print(f"Suggested Learning Rate: {lr_finder.suggestion()}")

    # Find optimal batch size
    #batch_size = tuner.scale_batch_size(model, datamodule=dm, mode="power")
    #print(f"Suggested Batch Size: {batch_size}")

    wandb.finish()

@click.command()
@click.option("--config", "-c", type=str, default="./config.toml")
@click.option("--subontology", "-s", type=str, default=None)
@click.option("--wandb_run_name", "-n", type=str, default=None)
@click.option("--go-release", "-r", type=str, default=None)
def tune(config, subontology, wandb_run_name, go_release):
    if ',' in subontology or ',' in go_release or ',' in config:
        matrix = itertools.product(subontology.split(','), go_release.split(','), config.split(','))
        for subontology, go_release, config in matrix:
            _tune(config, subontology, wandb_run_name, go_release)
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
    else:
        _tune(config, subontology, wandb_run_name, go_release)

if __name__ == "__main__":
    tune()
