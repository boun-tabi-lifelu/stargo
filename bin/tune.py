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
from data.datamodule import PUGODataModule

def _tune(config, subontology, wandb_run_name, go_release, data_dir):
    config_file_name = Path(config).name.split(".")[0]
    model_name = config_file_name.replace("_", "-")
    config = from_toml(config)

    if data_dir is None:
        data_dir = Path(config.train.data_dir)

    if 'pfresgo' not in str(data_dir):
        dataset_extra = "-" + data_dir.name
    else:
        dataset_extra = ""

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
        wandb_run_name = f"tune-stargo-{subontology_short}-{config.train.go_release}-{model_name}{dataset_extra}"

    logger = WandbLogger(project="stargo", name=wandb_run_name)

    # Define datamodule based on data directory
    if 'pugo' in str(data_dir) or 'deepgozero' in str(data_dir):
        dm = PUGODataModule(
            data_dir=data_dir,
            batch_size=config.train.batch_size,
            num_workers=config.train.dm_num_workers,
            ontology=config.train.subontology,
            go_release=config.train.go_release,
            order_go_terms=config.train.order_go_terms,
            train_go_embeddings=config.model.train_go_embeddings,
        )
    elif 'pfresgo' in str(data_dir):
        dm = PFresGODataModule(
            data_dir=data_dir,
            batch_size=config.train.batch_size,
            num_workers=config.train.dm_num_workers,
            ontology=config.train.subontology,
            go_release=config.train.go_release,
            order_go_terms=config.train.order_go_terms,
            train_go_embeddings=config.model.train_go_embeddings,
        )
    else:
        raise ValueError(f"Invalid data directory: {data_dir} - could not determine dataset type")

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
@click.option("--data-dir", "-d", type=Path, default=None, help="Use a custom data directory")
@click.option("--go-release", "-r", type=str, default=None)
@click.option("--subontology", "-s", type=str, default=None)
@click.option("--wandb_run_name", "-n", type=str, default=None)
def tune(config, subontology, wandb_run_name, go_release, data_dir):
    if ',' in subontology or ',' in go_release or ',' in config:
        matrix = itertools.product(subontology.split(','), go_release.split(','), config.split(','))
        for subontology, go_release, config in matrix:
            _tune(config, subontology, wandb_run_name, go_release, data_dir)
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
    else:
        _tune(config, subontology, wandb_run_name, go_release, data_dir)

if __name__ == "__main__":
    tune(auto_envvar_prefix="STARGO")
