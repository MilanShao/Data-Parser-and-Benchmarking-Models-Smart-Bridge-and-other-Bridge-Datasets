import copy
import json
import os.path
from pathlib import Path
from typing import Optional

import click
import wandb

from pbim_models.cli.util import load_config, prepare_trainer, _get_module_class
from pbim_models.dataset.dataset import WindowedDataModule


def load_parameter_config(config_file: Path):
    with open(config_file, "r") as f:
        return json.load(f)


def unify(config: dict, parameter_config: dict, unified_config: Optional[dict] = None):
    current = unified_config or copy.deepcopy(config)
    for key, value in parameter_config.items():
        if isinstance(value, dict) and key in config:
            unify(config[key], value, current[key])
        else:
            current[key] = value
    return current


def _restore_checkpoint(user: str, project: str, run_id: str) -> Optional[Path]:
    checkpoint_reference = f"{user}/{project}/MODEL-{run_id}:latest"
    # download checkpoint locally (if not already cached)
    try:
        artifact = wandb.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()
        return artifact_dir / "model.ckpt"
    except wandb.errors.CommError:
        return None


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "parameter_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def sweep(config_file: Path, parameter_file: Path):
    base_config = load_config(config_file)
    parameter_config = load_config(parameter_file)
    unified_config = unify(base_config, parameter_config)
    module_class = _get_module_class(unified_config)
    wandb.init(resume=True)
    wandb.save(os.path.join(wandb.run.dir, "config.yaml"))
    if wandb.run.resumed:
        print("Resuming previous run.")
        checkpoint_path = _restore_checkpoint(
            wandb.run.entity, wandb.run.project, wandb.run.id
        )
        if not checkpoint_path:
            print("Did not find a checkpoint to resume from. Starting from scratch.")
        model = (
            module_class.load_from_checkpoint(checkpoint_path)
            if checkpoint_path
            else module_class(unified_config)
        )
    else:
        model = module_class(unified_config)
    datamodule = WindowedDataModule(unified_config)
    trainer = prepare_trainer(unified_config)
    trainer.fit(model, datamodule=datamodule)
    wandb.finish()
