import abc
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

from pbim_models.cli.util import load_config
from pbim_models.dataset.model import DatasetMetadata


class Anomaly(abc.ABC):
    @staticmethod
    def _find_index(metadata: DatasetMetadata, channel: str) -> int:
        return metadata.channel_order.index(channel)

    @abc.abstractmethod
    def apply(
        self, metadata: DatasetMetadata, data: torch.Tensor, channel: str
    ) -> torch.Tensor:
        pass


class OffsetMeanAnomaly(Anomaly):
    def __init__(
        self,
        absolute_offset: Optional[float] = None,
        scaling_factor: Optional[float] = None,
    ):
        self._absolute_offset = absolute_offset
        self._scaling_factor = scaling_factor
        if absolute_offset is None and scaling_factor is None:
            raise ValueError(
                "Either absolute_offset or scaling_factor must be provided."
            )
        if absolute_offset is not None and scaling_factor is not None:
            raise ValueError(
                "Only one of absolute_offset or scaling_factor must be provided."
            )

    def apply(
        self, metadata: DatasetMetadata, data: torch.Tensor, channel: str
    ) -> None:
        index = self._find_index(metadata, channel)
        if self._scaling_factor is not None:
            data[..., index] *= self._scaling_factor
        else:
            data[..., index] += self._absolute_offset


class ScaleStdDevAnomaly(Anomaly):
    def __init__(self, scaling_factor: float):
        self._scaling_factor = scaling_factor

    def apply(
        self, metadata: DatasetMetadata, data: torch.Tensor, channel: str
    ) -> None:
        index = self._find_index(metadata, channel)
        mean = metadata.channel_stats[channel].mean
        data[..., index] = (data[..., index] - mean) * self._scaling_factor + mean


class WhiteNoiseAnomaly(Anomaly):
    def __init__(self, stddev: float):
        self._stddev = stddev

    def apply(
        self, metadata: DatasetMetadata, data: torch.Tensor, channel: str
    ) -> None:
        index = self._find_index(metadata, channel)
        data[..., index] += torch.randn_like(data[..., index]) * self._stddev


class SensorMalfunctionAnomaly(Anomaly):
    def __init__(self, malfunction_type: str):
        self._malfunction_type = malfunction_type

    def apply(
        self, metadata: DatasetMetadata, data: torch.Tensor, channel: str
    ) -> None:
        index = self._find_index(metadata, channel)
        match self._malfunction_type:
            case "zeros":
                data[..., index] = 0
            case "white_noise":
                data[..., index] = torch.randn_like(data[..., index])
            case _:
                raise ValueError(
                    f"Unknown malfunction type: '{self._malfunction_type}'"
                )


def build_anomaly(config: Dict[str, Any]) -> Anomaly:
    def get(name: str, default=None, allow_none=False):
        value = config.get(name, default)
        if value is None and not allow_none:
            raise ValueError(f"Missing value for key '{name}'")
        return value

    anomaly_type = get("type")
    match anomaly_type:
        case "offset_mean":
            return OffsetMeanAnomaly(
                absolute_offset=get("absolute_offset", allow_none=True),
                scaling_factor=get("scaling_factor", allow_none=True),
            )
        case "scale_std_dev":
            return ScaleStdDevAnomaly(scaling_factor=get("scaling_factor"))
        case "white_noise":
            return WhiteNoiseAnomaly(stddev=get("std_dev"))
        case _:
            raise ValueError(f"Unknown anomaly type '{anomaly_type}'.")


class AnomalyScenario:
    def __init__(
        self,
        name: str,
        anomalies: Dict[str, List[Anomaly]],
        description: Optional[str] = None,
        anomaly_probability=0.5,
    ):
        self._name = name
        self._anomalies = anomalies
        self._description = description
        self._anomaly_probability = anomaly_probability

    @property
    def name(self) -> str:
        return self._name

    @property
    def anomaly_probability(self) -> float:
        return self._anomaly_probability

    def by_channels(self) -> Dict[str, List[Anomaly]]:
        return self._anomalies

    @staticmethod
    def from_file(file_path: Path) -> "AnomalyScenario":
        config = load_config(file_path)
        return AnomalyScenario.from_config(config)

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "AnomalyScenario":
        name = config["name"]
        description = config.get("description", None)
        probability = config.get("anomaly_probability", 0.5)
        anomaly_config = config["anomalies"]
        if not anomaly_config:
            return AnomalyScenario(name, {}, description, probability)
        if "all" in anomaly_config:
            base_anomalies = [build_anomaly(a) for a in anomaly_config["all"]]
        else:
            base_anomalies = []
        anomalies = {
            channel: base_anomalies + [build_anomaly(a) for a in anomalies]
            for channel, anomalies in anomaly_config.items()
            if channel != "all"
        }
        return AnomalyScenario(name, anomalies, description, probability)
