import os
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple

import click
import numpy as np
import pytorch_lightning
import torch
import tqdm
import wandb


from pbim_models.cli.util import (
    load_config,
    download_data,
    sanitize_config,
    replace_placeholders,
)
from pbim_models.module.util import get_map_location
from pbim_models.dataset.dataset import WindowedDataModule
from pbim_models.module.module import TimeSeriesModule


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


def _move_to_device(batch: Dict[str, Any], device: str | torch.device):
    for key, value in batch.items():
        if isinstance(value, dict):
            _move_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _predict(
    prediction_config: Dict[str, Any], run_config: Dict[str, Any]
) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
    wandb.init(
        entity=run_config["entity"],
        project=run_config["project"],
        id=run_config["id"],
        resume="must",
    )
    data_dir = download_data(run_config)
    config = sanitize_config(
        load_config(data_dir / "config.yaml", from_wandb=True),
        overrides=prediction_config.get("overrides", None),
    )
    anomaly_config = prediction_config.get("anomaly_scenario", None)
    datamodule = WindowedDataModule(config, anomaly_config=anomaly_config)
    map_location = (
        config["trainer"]["accelerator"] if config["trainer"]["accelerator"] else None
    )
    model = TimeSeriesModule.load_from_checkpoint(
        data_dir / "model.ckpt",
        config=config,
    ).to(get_map_location(map_location))
    model.eval()
    datamodule.prepare_data()
    datamodule.setup(stage="predict")
    dataloader = datamodule.predict_dataloader()
    torch.set_grad_enabled(False)
    model.eval()
    predict_batches = config.get("trainer", {}).get(
        "limit_predict_batches", len(dataloader)
    )
    dataloader = datamodule.predict_dataloader()
    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader), total=predict_batches, desc="Predicting", leave=False
    ):
        if batch_idx >= predict_batches:
            break
        _move_to_device(batch, get_map_location(map_location))
        predictions = model.predict(batch, batch_idx)
        yield predictions["prediction"], predictions["target"]
    wandb.finish()


@click.command()
@click.argument("prediction_config_file", type=click.Path(exists=True, dir_okay=False))
def predict(prediction_config_file: Path):
    os.environ["WANDB_SILENT"] = "true"
    pytorch_lightning.seed_everything(42)
    prediction_config = load_config(prediction_config_file)
    run_config = prediction_config["run"]
    replace_placeholders(prediction_config, {"id": run_config["id"]})
    should_save_predictions = prediction_config.get("save_predictions", False)
    should_save_errors = prediction_config.get("save_errors", False)
    ids = run_config["id"]

    if ids is None or ids == "":
        ids = wandb.Api().runs(
            path=f"{run_config['entity']}/{run_config['project']}",
            filters={"state": "finished"},
        )
        ids = [run.id for run in ids]
    elif isinstance(ids, str):
        ids = [ids]
    for run_id in tqdm.tqdm(ids, desc="Runs"):
        this_run_config = run_config.copy()
        this_run_config["id"] = run_id
        output_path = Path(prediction_config["output_path"])
        output_path.mkdir(parents=True, exist_ok=True)
        predictions = []
        targets = []
        errors = []
        for prediction, target in _predict(prediction_config, this_run_config):
            prediction, target = prediction.cpu(), target.cpu()
            if should_save_predictions:
                predictions.append(prediction)
                targets.append(target)
            if should_save_errors:
                error = prediction - target
                errors.append(error)

        if should_save_predictions:
            predicted_tensors = torch.cat(predictions)
            target_tensors = torch.cat(targets)
            np.save(str(output_path / "predicted.npy"), predicted_tensors.numpy())
            np.save(str(output_path / "target.npy"), target_tensors.numpy())
        if should_save_errors:
            np.save(str(output_path / "errors.npy"), torch.cat(errors).numpy())
