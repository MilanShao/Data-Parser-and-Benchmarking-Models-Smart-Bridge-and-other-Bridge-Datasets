from pathlib import Path
from typing import Dict, Any, List

import click
import numpy as np
import pytorch_lightning
import torch
import wandb

from pbim_models.cli.util import (
    load_config,
    prepare_trainer,
    download_data,
    sanitize_config,
    save_metrics_single,
    save_metrics_multiple,
    _get_module_class,
    replace_placeholders,
)
from pbim_models.module.util import get_map_location
from pbim_models.dataset.dataset import WindowedDataModule


def _merge_results(
    new_metrics: Dict[Any, torch.Tensor], current: Dict[Any, List[float]]
) -> Dict[str, List[float]]:
    for key, value in new_metrics.items():
        if key not in current:
            current[key] = []
        current[key].append(value.item())
    return current


def _flatten_results(
    results: Dict[Any, Dict[Any, torch.Tensor]]
) -> Dict[Any, List[float]]:
    flattened = {}
    for key, value in results.items():
        flattened = _merge_results(value, flattened)
    return flattened


def _make_statistics(results: Dict[Any, List[float]]) -> Dict[Any, Dict[str, float]]:
    statistics = {}
    for key, value in results.items():
        statistics[key] = {
            "mean": np.mean(value).item(),
            "std": np.std(value).item(),
        }
    return statistics


def _evaluate(evaluation_config: Dict[str, Any], run_config: Dict[str, Any]):
    print(
        f"Evaluating '{run_config['id']}' in project {run_config['project']} of entity {run_config['entity']}."
    )
    wandb.init(
        entity=run_config["entity"],
        project=run_config["project"],
        id=run_config["id"],
        resume="must",
    )
    data_dir = download_data(run_config, use_best=run_config.get("use_best", True))
    config = sanitize_config(
        load_config(data_dir / "config.yaml", from_wandb=True),
        overrides=evaluation_config.get("overrides", None),
    )
    datamodule = WindowedDataModule(config)
    map_location = (
        config["trainer"]["accelerator"] if config["trainer"]["accelerator"] else None
    )
    model = _get_module_class(config).load_from_checkpoint(
        data_dir / "model.ckpt",
        map_location=get_map_location(map_location),
        config=config,
    )
    trainer = prepare_trainer(config, offline=True, is_test=True)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    return model.metrics["test_metrics"].compute()


def _evaluate_multiple(config: Dict[str, Any], ids: List[str], output_path: Path):
    run_config = config["run"]
    separate = run_config.get("separate", False)
    print(f"Evaluating multiple runs (separate: {separate}).")
    results = {}
    for i, run_id in enumerate(ids):
        print(f"Evaluating run {i + 1}/{len(ids)}.")
        current_run_config = run_config.copy()
        current_run_config["id"] = run_id
        replace_placeholders(config, {"id": run_id})
        results[run_id] = _evaluate(config, current_run_config)
        save_metrics_single(results[run_id], output_path / f"{run_id}.json")
    if not separate:
        save_metrics_multiple(
            _make_statistics(_flatten_results(results)),
            output_path / "summary.json",
        )


def retrieve_ids_for_project(project: str, entity: str):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return [run.id for run in runs if run.state == "finished"]


@click.command()
@click.argument("evaluation_config_file", type=click.Path(exists=True, dir_okay=False))
def evaluate(evaluation_config_file: Path):
    pytorch_lightning.seed_everything(42)
    evaluation_config = load_config(evaluation_config_file)
    run_config = evaluation_config["run"]
    output_path = Path(evaluation_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    if run_config["id"] == "":
        run_config["id"] = None
    if isinstance(run_config["id"], list):
        _evaluate_multiple(evaluation_config, run_config["id"], output_path)
    elif isinstance(run_config["id"], str):
        run_id = run_config["id"]
        print("Evaluating single run.")
        replace_placeholders(evaluation_config, {"id": run_id})
        results = _evaluate(evaluation_config, run_config)
        save_metrics_single(results, output_path / f"{run_id}.json")
    elif run_config["id"] is None:
        print("No run id specified. Evaluating all runs in the project.")
        ids = retrieve_ids_for_project(run_config["project"], run_config["entity"])
        _evaluate_multiple(evaluation_config, ids, output_path)
    else:
        raise ValueError(f"Invalid run id: {run_config['id']}")
