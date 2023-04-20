import random
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional

import torch

from pbim_models.dataset.anomaly import Anomaly
from pbim_models.dataset.model import DatasetMetadata

Transform = Callable[[DatasetMetadata, Any, int], Any]


class Transforms:
    @staticmethod
    def default_predicate(channel: str) -> bool:
        return channel != "Time"

    @staticmethod
    def identity_transform(metadata: DatasetMetadata, sample: Any, index: int) -> Any:
        return sample

    @staticmethod
    def to_tensor_transform(
        metadata: DatasetMetadata, sample: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        data = sample["data"]
        sample["data"] = torch.tensor(
            [data[channel] for channel in metadata.channel_order if channel in data]
        )
        return sample

    class FilterTransform:
        def __init__(self, predicate: Callable[[str], bool]):
            self._predicate = predicate

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            channel_order = sample.get("_channels", metadata.channel_order)
            data = sample["data"]
            sample["data"] = {
                channel: data[channel]
                for channel in channel_order
                if self._predicate(channel)
            }
            sample["_channel_indices"] = [
                i for i, channel in enumerate(channel_order) if self._predicate(channel)
            ]
            sample["_channels"] = [
                channel for channel in channel_order if self._predicate(channel)
            ]
            return sample

        def __call__(
            self, metadata: DatasetMetadata, data: Dict[str, Any], index: int
        ) -> Any:
            return self._transform(metadata, data, index)

    class ZScoreTransform:
        def __init__(self, predicate: Optional[Callable[[str], bool]] = None):
            self._predicate = predicate or Transforms.default_predicate

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            data = sample["data"]
            for i, channel in enumerate(
                [
                    channel
                    for channel in metadata.channel_order
                    if self._predicate(channel)
                ]
            ):
                std, mean = (
                    metadata.statistics[channel].std,
                    metadata.statistics[channel].mean,
                )
                if std == 0:
                    std = 1
                data[..., i] = (data[..., i] - mean) / std
            return sample

        def __call__(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Any:
            return self._transform(metadata, sample, index)

    class MinMaxTransform:
        def __init__(self, predicate: Optional[Callable[[str], bool]] = None):
            self._predicate = predicate or Transforms.default_predicate

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            data = sample["data"]
            for i, channel in enumerate(metadata.channel_order):
                if self._predicate(channel):
                    data[i] = (data[i] - metadata.statistics[channel].min) / (
                        metadata.statistics[channel].max
                        - metadata.statistics[channel].min
                    )
            return sample

        def __call__(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ):
            return self._transform(metadata, sample, index)

    class IntroduceAnomalyTransform:
        def __init__(
            self,
            anomalies: Dict[str, Anomaly | List[Anomaly]],
            anomaly_probability: float,
        ):
            self._anomalies = anomalies
            self._anomaly_probability = anomaly_probability

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            if random.uniform(0, 1) > self._anomaly_probability:
                sample["anomaly"] = False
                return sample
            data = sample["data"]
            for channel, anomalies in self._anomalies.items():
                if not isinstance(anomalies, list):
                    anomalies = [anomalies]
                for anomaly in anomalies:
                    anomaly.apply(metadata, data, channel)
            sample["anomaly"] = True
            return sample

        def __call__(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ):
            return self._transform(metadata, sample, index)

    class LabelAnomalyTransform:
        def __init__(self, anomaly_index_path: Path):
            self._index = self._make_index(anomaly_index_path)

        @staticmethod
        def _make_index(anomaly_index_path: Path) -> Dict[int, bool]:
            index = {}
            with open(anomaly_index_path, "r") as f:
                for line in f:
                    index, label = line.split()
                    index[int(index)] = bool(label)
            return index

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            sample["anomaly"] = self._index[index]
            return sample

        def __call__(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ):
            return self._transform(metadata, sample, index)

    class GroupChannelsTransform:
        def __init__(
            self,
            grouping_profile: str,
            additional_channels_per_node_groups: Dict[str, List[str]],
        ):
            self._grouping_profile = grouping_profile
            self._additional_channels_per_node_groups = (
                additional_channels_per_node_groups
            )

        def _transform(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            additional_channels_per_node = self._additional_channels_per_node_groups[
                self._grouping_profile
            ]
            # if the additional channels are not present, do nothing
            if not all(
                channel in sample["_channels"]
                for channel in additional_channels_per_node
            ):
                raise ValueError(
                    f"Additional channels {additional_channels_per_node} not found in sample."
                )
            if len(additional_channels_per_node) == 0:
                return sample
            data = sample["data"]
            indices = [
                sample["_channels"].index(v) for v in additional_channels_per_node
            ]
            node_data = torch.unbind(data, dim=-1)
            additional_features = torch.cat(
                [node_data[i].unsqueeze(-1) for i in indices], dim=-1
            )
            data = [
                torch.cat([node_data[i].unsqueeze(-1), additional_features], dim=-1)
                for i in range(len(node_data))
                if i not in indices
            ]
            sample["data"] = torch.stack(data, dim=1)
            sample["_channels"] = [
                channel
                for channel in sample["_channels"]
                if channel not in additional_channels_per_node
            ]
            sample["_channel_indices"] = [
                i
                for i, channel in zip(sample["_channel_indices"], sample["_channels"])
                if channel not in additional_channels_per_node
            ]
            return sample

        def __call__(
            self, metadata: DatasetMetadata, sample: Dict[str, Any], index: int
        ):
            return self._transform(metadata, sample, index)

    class ComposeTransforms:
        def __init__(self, *transforms: Transform):
            self._transforms = transforms

        def _transform(
            self, metadata: DatasetMetadata, data: Dict[str, Any], index: int
        ) -> Any:
            for transform in self._transforms:
                data = transform(metadata, data, index)
            return data

        def __call__(self, metadata: DatasetMetadata, data: Dict[str, Any], index: int):
            return self._transform(metadata, data, index)
