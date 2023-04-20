import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Type

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)

from pbim_models.models.unsupervised.agcrn.net import AGCRN
from pbim_models.models.unsupervised.lstm.net import LSTMModel
from pbim_models.models.unsupervised.mtgnn.net import MtGNN
from pbim_models.models.unsupervised.stemgnn.net import StemGNN
from pbim_models.models.unsupervised.tcn.net import TCN
from pbim_models.models.util import LossWrapper
from pbim_models.models.unsupervised.vrae.net import VRAE
from pbim_models.module.module import TimeSeriesModule, TimeSeriesGANModule

METRICS = [
    MeanAbsoluteError(),
    MeanSquaredError(squared=False),
    MeanAbsolutePercentageError(),
]


def load_config(config_path: Path, from_wandb: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "base" not in config or from_wandb:
        return config

    base_config_path = config["base"]
    with open(config_path.parent / base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    return merge_configs(base_config, config)


def prepare_loss(config: Dict[str, Any]):
    loss_config = config["loss"]
    name = loss_config["type"]
    reduction = loss_config.get("reduction", "mean")
    multi_step = loss_config.get("multi_step", False)
    return LossWrapper(
        name=name,
        reduction=reduction,
        multi_step=multi_step,
    )


def prepare_trainer(
    config: Dict[str, Any],
    offline: bool = False,
    is_test: bool = False,
    additional_callbacks: Optional[List[Callback]] = None,
):
    trainer_config = config["trainer"]
    checkpoint_config = trainer_config.get("checkpoint", None)
    early_stopping_config = trainer_config.get("early_stopping", None)
    logger_config = trainer_config["logger"]
    callbacks = []

    def get(
        name: str,
        config: Dict[str, Any] = trainer_config,
        default=None,
        allow_none=False,
    ):
        value = config.get(name, default)
        if value is None and not allow_none:
            raise ValueError(f"Missing value for key '{name}' in config.")
        return value

    if checkpoint_config is not None:
        checkpoint = ModelCheckpoint(
            monitor=get(
                "metric", config=checkpoint_config, default=None, allow_none=True
            ),
            dirpath=get("dirpath", config=checkpoint_config),
            save_top_k=get("save_top_k", config=checkpoint_config, default=0),
            filename=get(
                "filename", config=checkpoint_config, default=None, allow_none=True
            ),
        )
        callbacks.append(checkpoint)

    if early_stopping_config is not None:
        early_stopping = EarlyStopping(
            monitor=get("metric", config=early_stopping_config),
            patience=get("patience", config=early_stopping_config),
            verbose=get("verbose", config=early_stopping_config, default=False),
            min_delta=get("delta", config=early_stopping_config, default=0.0),
            mode=get("mode", config=early_stopping_config, default="min"),
        )
        callbacks.append(early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    if not is_test:
        logger = WandbLogger(
            log_model=not offline
            and get("log_model", config=logger_config, default=False),
            project=get("project", config=logger_config, allow_none=True),
            name=get("name", config=logger_config, allow_none=True),
            offline=offline,
        )
        logger.experiment.config.update(config)
    else:
        logger = None
    return Trainer(
        logger=not is_test and logger,
        devices=get("devices", default="auto"),
        accelerator=get("accelerator", default="cpu"),
        gradient_clip_val=get("clip_grad_norm", allow_none=True),
        max_epochs=get("max_epochs", allow_none=True, default=None),
        limit_test_batches=get("limit_test_batches", default=1.0),
        limit_val_batches=get("limit_val_batches", default=1.0),
        limit_train_batches=get("limit_train_batches", default=1.0),
        callbacks=callbacks + (additional_callbacks or []),
    )


MODEL_CLASSES = {
    "agcrn": AGCRN,
    "lstm": LSTMModel,
    "mtgnn": MtGNN,
    "stemgnn": StemGNN,
    "tcn": TCN,
    "vrae": VRAE,
}


def load_model(model_type: str, checkpoint_file: Path):
    model_cls = MODEL_CLASSES.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model type '{model_type}'.")
    return model_cls.load_from_checkpoint(checkpoint_file)


def save_metrics_single(metrics: Dict[str, List[float]], output_file: Path):
    values = {
        metric: value.item() if isinstance(value, torch.Tensor) else value
        for metric, value in metrics.items()
    }
    results = {"metrics": values}
    with open(output_file, "w") as f:
        json.dump(results, f)


def save_metrics_multiple(metrics: Dict[str, Dict[str, float]], output_file: Path):
    results = {"metrics": metrics}
    with open(output_file, "w") as f:
        json.dump(results, f)


def download_data(
    run_config: Dict[str, Any], output_dir: Path = Path("/tmp"), use_best: bool = True
):
    run_id = run_config["id"]
    suffix = "best_k" if use_best else "best"  # best actually means last
    wandb.restore("config.yaml", root=str(output_dir), replace=True)
    artifact_id = run_config.get("artifact_id", f"model-{run_id}:{suffix}")
    model_artifact = wandb.use_artifact(artifact_id, type="model")
    model_artifact.download(root=str(output_dir))
    return output_dir


def merge_configs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in right.items():
        if k in left and isinstance(left[k], dict) and isinstance(right[k], dict):
            merge_configs(left[k], right[k])
        else:
            left[k] = right[k]

    return left


def _build_from_separated_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a nested config from a flat config with keys separated by '/'."""
    nested_config = {}
    for key, value in config.items():
        parts = key.split("/")
        current = nested_config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested_config


def sanitize_config(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            if "value" in value:
                return value["value"]
            return {
                k: _sanitize(v)
                for k, v in value.items()
                if not k.startswith("_") and not k.startswith("wandb") and not "/" in k
            }
        elif isinstance(value, list):
            return [_sanitize(v) for v in value]
        return value

    sanitized = _sanitize(config)
    # Handle "/" separated keys.
    slash_keys = set([key.split("/")[0] for key in config.keys() if "/" in key])
    for slash_key in slash_keys:
        sub_config = _sanitize(
            _build_from_separated_keys(
                {
                    key: value
                    for key, value in config.items()
                    if key.startswith(f"{slash_key}/")
                }
            )
        )
        if slash_key in sanitized:
            sanitized = merge_configs(sanitized, sub_config)
        else:
            sanitized[slash_key] = sub_config

    if overrides:
        sanitized = merge_configs(sanitized, overrides)
    return sanitized


def _get_module_class(config: Dict[str, Any]) -> Type[pl.LightningModule]:
    module_type = config.get("type", "standard")
    match module_type:
        case "standard":
            return TimeSeriesModule
        case "gan":
            return TimeSeriesGANModule
        case _:
            raise ValueError(f"Unknown model type: {module_type}")


def _get_module(config: Dict[str, Any]) -> pl.LightningModule:
    return _get_module_class(config)(config)


def _replace(value: str, overrides: Dict[str, Any]):
    for match in re.findall(r"\$\w+", value):
        value = value.replace(match, str(overrides[match[1:].lower()]))
    return value


def replace_placeholders(config: Dict[str, Any], overrides: Dict[str, Any]):
    for key, value in config.items():
        if isinstance(value, dict):
            replace_placeholders(value, overrides)
        elif isinstance(value, str):
            if "$" in value:
                config[key] = _replace(value, overrides)
