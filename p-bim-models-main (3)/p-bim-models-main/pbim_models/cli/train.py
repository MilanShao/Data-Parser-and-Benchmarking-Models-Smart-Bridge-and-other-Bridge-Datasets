from pathlib import Path

import click
import wandb

from pbim_models.cli.util import load_config, prepare_trainer, _get_module
from pbim_models.dataset.dataset import WindowedDataModule


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--offline", is_flag=True, default=False)
def train(config_file: Path, offline: bool):
    config = load_config(config_file)
    datamodule = WindowedDataModule(config)
    module = _get_module(config)
    trainer = prepare_trainer(config, offline=offline)
    trainer.fit(module, datamodule=datamodule)
    wandb.finish()
