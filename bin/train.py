import sys

sys.path.append(".")
import gc
import os
import itertools
from pathlib import Path

import click
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

import wandb
from config import from_toml
from data.datamodule import PFresGODataModule, PUGODataModule
from model import get_model

try:
  from IPython.core import ultratb
  sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=1)
except ImportError:
  print("IPython not found, using default exception hook")
  def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

def _main(config, subontology, wandb_run_name, go_release, checkpoint, dry_run, compile, data_dir):
  config_file_name = Path(config).name.split(".")[0]
  model_name = config_file_name.replace("_", "-")
  config = from_toml(config)
  if data_dir is None:
    data_dir = Path(config.train.data_dir)

  if 'pfresgo' not in str(data_dir):
    dataset_extra = "-" + data_dir.name
  else:
    dataset_extra = ""


  # Apply overrides
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

  model_id = f"contempro-{subontology_short}-{config.train.go_release}-{model_name}{dataset_extra}"

  if wandb_run_name is not None:
    print("Overriding wandb run name to", wandb_run_name)
    wandb_run_name = wandb_run_name
  else:
    wandb_run_name = model_id

  # Return config to be used if dry run
  if dry_run:
    print(f"Wandb run name: {wandb_run_name}")
    from pprint import pprint
    pprint(config)
    return

  offline = wandb.api.api.settings().get("mode") == "offline"

  # Define logger: TensorBoardLogger if WanDB is offline else WandbLogger
  if offline:
    logger = TensorBoardLogger(save_dir="logs", name=wandb_run_name)
  else:
    logger = WandbLogger(project="contempro", log_model="all", name=wandb_run_name, checkpoint_name=model_id)

    # logger.experiment.config.update({
    #   "model": config.model.__dict__,
    #   "train": config.train.__dict__
    # })

  # Define datamodule
  if 'pugo' in str(data_dir) or 'netgo' in str(data_dir):
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

  # If training embeddings, setup the datamodule to get determine go_input_dim == vocab size
  if config.model.train_go_embeddings:
    dm.setup("fit")

    # override the go_input_dim to be the number of GO terms if training embeddings, in this case it's not the dimension but the vocab size
    config.model.go_input_dim = len(dm.train_dataset.go_term_list)

  # Prepare data (not implemented)
  if config.train.prepare_data:
    dm.prepare_data()

  # Create model
  model = get_model(config.model.name, config)

  # torch.compile optimization (not tested)
  if compile:
    # user directory is not executable, so we use /tmp
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.makedirs("/tmp/triton_cache", exist_ok=True)
    model = torch.compile(model)

  # Checkpoint loading from WanDB
  checkpoint_name = model_id

  if checkpoint:
    try:
      # Download the artifact
      if wandb.run is not None:
        artifact = wandb.run.use_artifact(checkpoint_name+":latest")
        artifact_dir = artifact.download()
      if wandb.run is None:
        api = wandb.Api()
        artifact = api.artifact(f"contempro/{checkpoint_name}:best")
        artifact_dir = artifact.download()

      # Find the checkpoint file
      checkpoint_path = list(Path(artifact_dir).rglob("*.ckpt"))[0]

      # Load the checkpoint
      print(f"Found checkpoint from {checkpoint}")
    except Exception as e:
      print(f"Error loading checkpoint: {e}. Continuing without checkpoint.")
      checkpoint_path = None
  else:
    checkpoint_path = None

  # Training callbacks
  callbacks = [ModelCheckpoint(monitor="val_loss", mode="min", filename=checkpoint_name)]

  if config.train.early_stopping is not None:
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=config.train.early_stopping))

  # Trainer setup
  precision = "bf16-true" if config.train.use_tpu else "bf16-mixed" if config.train.bf16_precision else 32
  trainer = Trainer(
    max_epochs=config.train.max_epochs,
    callbacks=callbacks,
    logger=[logger],
    devices=3, # todo
    strategy="fsdp",
    accelerator="tpu" if config.train.use_tpu else "auto",
    accumulate_grad_batches=config.train.gradient_accumulation if config.train.gradient_accumulation is not None else 1,
    precision=precision
  )

  # Train
  print(f"Loading checkpoint from {checkpoint_path}")
  trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path if checkpoint_path is not None else None)

  # Finish WanDB run (for other iterations to start new ones)
  wandb.finish()


@click.command()
@click.option("--config", "-c", type=str, default="./config.toml")
@click.option("--subontology", "-s", type=str, default=None, help="Override the subontology to use for training")
@click.option("--wandb_run_name", "-n", type=str, default=None, help="Override the wandb run name, by default it'll be contempro-<subontology>-<go_release>-<model_name>. Eg. contempro-bp-2020-ordered-encdec-medium")
@click.option("--go-release", "-r", type=str, default=None, help="Override the GO release to use for training")
@click.option("--checkpoint", type=str, default=None, help="Use checkpoint from wandb", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Don't train but only print the final configs")
@click.option("--compile", is_flag=True, help="Compile the model")
@click.option("--data-dir", type=str, default=None, help="Use a custom data directory")
def main(config, subontology, wandb_run_name, go_release, checkpoint, dry_run, compile, data_dir):
  if data_dir is not None:
    data_dir = Path(data_dir)

  if ',' in subontology or ',' in go_release or ',' in config:
    matrix = itertools.product(subontology.split(','), go_release.split(','), config.split(','))
    for subontology, go_release, config in matrix:
      _main(config, subontology, wandb_run_name, go_release, checkpoint, dry_run, compile, data_dir)
      import gc
      gc.collect()
      torch.cuda.empty_cache()
      gc.collect()

  else:
    _main(config, subontology, wandb_run_name, go_release, checkpoint, dry_run, compile, data_dir)

if __name__ == "__main__":
  main()
