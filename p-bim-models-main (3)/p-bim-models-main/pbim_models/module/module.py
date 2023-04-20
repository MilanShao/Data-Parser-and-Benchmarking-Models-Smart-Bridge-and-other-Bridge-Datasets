import copy
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from scipy import linalg
from scipy.spatial.distance import mahalanobis
from torch import nn
from torch.optim import Optimizer

from pbim_models.metrics.metric import TimeSeriesMetric
from pbim_models.models.gan.dcgan.layer import Discriminator, ConvolutionalGenerator
from pbim_models.models.util import Stage
from pbim_models.module.util import (
    build_model,
    build_loss,
    build_metrics,
    build_optimizers,
    denormalize,
    malahanobis_distance,
)

import matplotlib.pyplot as plt

from pbim_models.util import aggregate_errors, compute_mean_and_covariance


class TimeSeriesModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._config = config
        self._model = self._build_model()
        self._use_model_loss = config.get("use_model_loss", False)
        self._loss = self._build_loss() if not self._use_model_loss else None
        self._metrics = self._build_metrics()
        self._optimizers = self._build_optimizers()
        self._should_denormalize = config.get("denormalize", True)
        self._is_supervised = config.get("supervised", False)
        self._test_config = None

        self.save_hyperparameters(config)

    @property
    def metrics(self) -> nn.ModuleDict:
        return self._metrics

    def _build_model(self):
        return build_model(self._config)

    def _build_loss(self) -> nn.Module:
        return build_loss(self._config)

    def _build_metrics(self) -> nn.ModuleDict:
        return build_metrics(self._config)

    def _build_optimizers(self):
        return build_optimizers(self._config, self._model)

    def _get_batch_size(self, stage: Stage) -> int:
        return (
            self._config["dataset"]
            .get(stage, {})
            .get("batch_size", self._config["dataset"]["base"]["batch_size"])
        )

    def _verify_batch_size(self, batch: Dict[str, Any], stage: Stage):
        batch_size = self._get_batch_size(stage)
        data = batch["train"]["data"]
        if data.shape[0] != batch_size:
            raise ValueError(
                f"Expected batch size {batch_size} but got {data.shape[0]}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["train"]["data"]
        if not self._is_supervised:
            y = batch["target"]["data"]
        else:
            y = batch["target"]["anomaly"]
        # Shape B x T x N  -> B x T x N x (C = 1)
        x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        y_hat = self(x)
        if self._use_model_loss:
            loss = self._model.compute_loss(y_hat, y)
        else:
            loss = self._loss(y_hat, y)
        if self._should_denormalize:
            channel_indices = batch["train"]["_channel_indices"]
            metadata = batch["_metadata"]
            if not self._is_supervised:
                y = denormalize(y, channel_indices, metadata)
                y_hat = denormalize(y_hat, channel_indices, metadata)
        return loss, y, y_hat

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        self._verify_batch_size(batch, "train")
        loss, y, y_hat = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        self._update_metrics("train", y, y_hat)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._verify_batch_size(batch, "val")
        loss, y, y_hat = self.step(batch, batch_idx)
        self.log("val_loss", loss)
        self._update_metrics("val", y, y_hat)
        return loss

    def on_test_start(self) -> None:
        if "test_config" not in self._config:
            raise ValueError("No test config found in config file")
        test_config = self._config["test_config"]
        error_aggregation = test_config["aggregation"]
        errors = np.abs(np.load(test_config["errors"]))
        errors = aggregate_errors(error_aggregation, errors)
        mean, cov = compute_mean_and_covariance(errors)
        flattened_errors = errors[:, :, 0].reshape(-1, errors.shape[-2])
        specific_params = {}
        match test_config["type"]:
            case "mahalanobis":
                inv_cov = linalg.pinv(cov)
                distances = np.array(
                    [mahalanobis(u, mean, inv_cov) for u in flattened_errors]
                )
                specific_params = {
                    "icov": torch.from_numpy(inv_cov).to(
                        device=self.device, dtype=torch.float
                    )
                }
            case "mae":
                distances = np.mean(flattened_errors, axis=1)
            case _:
                raise ValueError(f"Unknown test config type {test_config['type']}")
        threshold_fraction = test_config["threshold_fraction"]
        threshold = np.quantile(distances, threshold_fraction)
        self._test_config = {
            "type": test_config["type"],
            "mean": torch.from_numpy(mean).to(device=self.device, dtype=torch.float),
            "threshold_distance": threshold.item(),
            "aggregation": error_aggregation,
            **specific_params,
        }

    def _compute_distance(self, y: torch.Tensor, y_hat: torch.Tensor):
        if self._test_config is None:
            raise ValueError("Test config not initialized")
        error_aggregation = self._test_config["aggregation"]
        errors = torch.abs(y - y_hat)
        errors = aggregate_errors(error_aggregation, errors)
        match self._test_config["type"]:
            case "mahalanobis":
                distances = malahanobis_distance(
                    errors[:, :, 0],
                    self._test_config["mean"],
                    self._test_config["icov"],
                )
            case "mae":
                distances = torch.mean(errors[:, :, 0], dim=1)
            case _:
                raise ValueError(
                    f"Unknown test config type {self._test_config['type']}"
                )
        return distances

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        if self._test_config is None:
            raise ValueError("Test config not initialized")
        self._verify_batch_size(batch, "test")
        loss, y, y_hat = self.step(batch, batch_idx)
        threshold = self._test_config["threshold_distance"]
        distances = self._compute_distance(y, y_hat)
        predictions = torch.where(distances > threshold, 1.0, 0.0)
        self.log("test_loss", loss)
        self._update_metrics("test", batch["target"].get("anomaly"), predictions)
        return loss

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        loss, y, y_hat = self.step(batch, batch_idx)
        return {
            "target": y,
            "prediction": y_hat,
        }

    def predict(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        return self.predict_step(batch, batch_idx, dataloader_idx)

    def _update_metrics(
        self,
        stage: Stage,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ):
        metrics = self._metrics[f"{stage}_metrics"]
        for metric_name, metric in metrics.items():
            if (
                isinstance(metric, TimeSeriesMetric)
                and metric.is_classification_metric()
            ):
                if self._is_supervised:
                    metric.update(y_hat.squeeze(1), y.squeeze(1))
                else:
                    metric.update(y_hat, y)
                if metric.returns_dict():
                    metric_values_with_stage_prefix = {
                        f"{stage}_{name}": value
                        for name, value in metric.compute().items()
                    }
                    self.log_dict(
                        metric_values_with_stage_prefix, on_step=False, on_epoch=True
                    )
                else:
                    self.log(
                        f"{stage}_{metric_name}", metric, on_step=False, on_epoch=True
                    )
            else:
                self.log(f"{stage}_{metric_name}", metric(y_hat, y))

    def configure_optimizers(self):
        return self._optimizers


class TimeSeriesGANModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._config = config
        self._generator = self._build_generator()
        self._discriminator = self._build_discriminator()
        self._loss = self._config["model"].get("loss", "bce")
        self._clip_value = self._config["model"]["discriminator"].get(
            "clip_value", 0.01
        )
        self._metrics = build_metrics(self._config)

        self.save_hyperparameters(config)

    @property
    def metrics(self) -> nn.ModuleDict:
        return self._metrics

    def _build_generator(self):
        generator_config = self._config["model"]["generator"]
        return ConvolutionalGenerator(
            layers=generator_config["layers"],
            noise_channels=generator_config["noise_channels"],
            noise_length=generator_config["noise_length"],
            noise_dimension=generator_config["input_size"],
            kernel_size=generator_config.get("kernel_size", None),
            window_size=self._config["dataset"]["base"]["window_size"],
            input_size=generator_config["input_size"],
            output_size=generator_config["output_size"],
            output_channels=generator_config["output_channels"],
        )

    def _build_generator_optimizer(self):
        generator_optimizer_config = copy.deepcopy(self._config)
        generator_optimizer_config["optimizers"] = generator_optimizer_config[
            "optimizers"
        ]["generator"]
        return build_optimizers(generator_optimizer_config, self._generator)

    def _build_discriminator(self):
        discriminator_config = self._config["model"]["discriminator"]
        return Discriminator(
            layers=discriminator_config["layers"],
            kernel_size=discriminator_config["kernel_size"],
            input_channels=discriminator_config["input_channels"],
            input_size=discriminator_config["input_size"],
            window_size=self._config["dataset"]["base"]["window_size"],
        )

    def _build_discriminator_optimizer(self):
        discriminator_optimizer_config = copy.deepcopy(self._config)
        discriminator_optimizer_config["optimizers"] = discriminator_optimizer_config[
            "optimizers"
        ]["discriminator"]
        return build_optimizers(discriminator_optimizer_config, self._discriminator)

    def _make_noise(self, batch_size: int):
        num_nodes = self._config["model"]["generator"]["input_size"]
        noise_channels = self._config["model"]["generator"]["noise_channels"]
        noise_length = self._config["model"]["generator"]["noise_length"]
        return torch.randn(batch_size, noise_length, num_nodes, noise_channels).to(
            self.device
        )

    def _generator_loss(self, logits: torch.Tensor):
        match self._loss:
            case "bce":
                return nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)
                )
            case "wasserstein":
                return -torch.mean(logits)
            case _:
                raise ValueError(f"Unknown loss {self._loss}")

    def _discriminator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):
        match self._loss:
            case "bce":
                return (
                    nn.functional.binary_cross_entropy_with_logits(
                        logits_real, torch.ones_like(logits_real)
                    )
                    + nn.functional.binary_cross_entropy_with_logits(
                        logits_fake, torch.zeros_like(logits_fake)
                    )
                ) / 2
            case "wasserstein":
                return -torch.mean(logits_real) + torch.mean(logits_fake)
            case _:
                raise ValueError(f"Unknown loss {self._loss}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._discriminator(x)

    def predict(self, batch: torch.Tensor | None, batch_idx: int):
        return self.predict_step(batch, batch_idx)

    def training_step(self, batch: Dict[str, Any], batch_idx: int, optimizer_idx: int):
        x = batch["train"]["data"]
        batch_size = x.shape[0]
        noise = self._make_noise(batch_size)
        if optimizer_idx == 0:
            # Train generator
            generated = self._generator(noise)
            loss = self._generator_loss(self._discriminator(generated))
            self.log("train_generator_loss", loss, prog_bar=True)
            return loss
        elif optimizer_idx == 1:
            # Train discriminator
            real = self._discriminator(x)
            fake = self._discriminator(self._generator(noise))
            loss = self._discriminator_loss(real, fake)
            self.log("train_discriminator_loss", loss, prog_bar=True)
            return loss

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        if self._loss == "wasserstein":
            for p in self._discriminator.parameters():
                p.data.clamp_(-self._clip_value, self._clip_value)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["train"]["data"]
        logits = self._discriminator(x)
        probs = 1 - torch.sigmoid(logits)
        targets = batch["train"]["anomaly"].to(torch.float)
        self._update_metrics("val", probs, targets)
        loss = nn.functional.binary_cross_entropy(
            probs, targets.unsqueeze(1).to(torch.float)
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["train"]["data"]
        logits = self._discriminator(x)
        probs = 1 - torch.sigmoid(logits)  # Convert to fake probability
        targets = batch["train"]["anomaly"].to(torch.float)
        self._update_metrics("test", probs, targets)
        loss = nn.functional.binary_cross_entropy(
            probs, targets.unsqueeze(1).to(torch.float)
        )
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(
        self, batch: torch.Tensor | None, batch_idx: int, dataloader_idx: int = 0
    ):
        noise = batch if batch is not None else self._make_noise(1)
        return self._generator(noise)

    def on_train_epoch_end(self) -> None:
        if not self._config.get("save_generated", False):
            return
        batch_size = 1
        noise = self._make_noise(batch_size)
        with torch.no_grad():
            self._generator.eval()
            generated = self._generator(noise).squeeze(-1).detach().cpu().numpy()
            self._generator.train()

        fig, ax = plt.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                ax[i, j].plot(generated[0, :, i * 4 + j])
        fig.tight_layout()
        fig.savefig(f"/tmp/generated-epoch-{self.current_epoch}.png")

    def _update_metrics(
        self,
        stage: Stage,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        metrics = self._metrics[f"{stage}_metrics"]
        for metric_name, metric in metrics.items():
            metric.update(predictions, targets)
            if metric.returns_dict():
                metric_values_with_stage_prefix = {
                    f"{stage}_{name}": value for name, value in metric.compute().items()
                }
                self.log_dict(
                    metric_values_with_stage_prefix, on_step=False, on_epoch=True
                )
            else:
                self.log(
                    f"{stage}_{metric_name}",
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        generator_optimizers_and_schedulers = self._build_generator_optimizer()
        if isinstance(generator_optimizers_and_schedulers, tuple):
            (
                generator_optimizers,
                generator_schedulers,
            ) = generator_optimizers_and_schedulers
            optimizers.extend(generator_optimizers)
            schedulers.extend(generator_schedulers)
        else:
            optimizers.extend(generator_optimizers_and_schedulers)

        discriminator_optimizers_and_schedulers = self._build_discriminator_optimizer()
        if isinstance(discriminator_optimizers_and_schedulers, tuple):
            (
                discriminator_optimizers,
                discriminator_schedulers,
            ) = discriminator_optimizers_and_schedulers
            optimizers.extend(discriminator_optimizers)
            schedulers.extend(discriminator_schedulers)
        else:
            optimizers.extend(discriminator_optimizers_and_schedulers)

        return optimizers, schedulers
