from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple

import click
import numpy as np
import pytorch_lightning
import torch
import tqdm
import wandb
from matplotlib import pyplot as plt

from pbim_models.cli.util import (
    load_config,
    prepare_trainer,
    download_data,
    sanitize_config,
    save_metrics_single,
    save_metrics_multiple,
)
from pbim_models.module.util import get_map_location
from pbim_models.dataset.dataset import WindowedDataModule
from pbim_models.module.module import TimeSeriesModule, TimeSeriesGANModule


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


def _predict(
    prediction_config: Dict[str, Any], run_config: Dict[str, Any]
) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
    wandb.init(
        entity=run_config["entity"],
        project=run_config["project"],
        id=run_config["id"],
        resume="must",
    )
    data_dir = download_data(run_config, use_best=False)
    config = sanitize_config(
        load_config(data_dir / "config.yaml", from_wandb=True),
        overrides=prediction_config.get("overrides", None),
    )
    map_location = (
        config["trainer"]["accelerator"] if config["trainer"]["accelerator"] else None
    )
    model = TimeSeriesGANModule.load_from_checkpoint(
        data_dir / "model.ckpt",
        map_location=get_map_location(map_location),
        config=config,
    )
    torch.set_grad_enabled(False)
    model.eval()
    predict_batches = config.get("trainer", {}).get("limit_predict_batches", 10)

    for batch_idx in tqdm.trange(predict_batches):
        predictions = model.predict(None, batch_idx)
        yield predictions
    wandb.finish()


@click.command()
@click.argument("prediction_config_file", type=click.Path(exists=True, dir_okay=False))
def predict_gan(prediction_config_file: Path):
    pytorch_lightning.seed_everything(42)
    prediction_config = load_config(prediction_config_file)
    run_config = prediction_config["run"]
    print("Predicting...")
    for i, prediction in enumerate(_predict(prediction_config, run_config)):
        fig, ax = plt.subplots()
        ax.plot(prediction[0][:, 10].detach().numpy())
        fig.show()
