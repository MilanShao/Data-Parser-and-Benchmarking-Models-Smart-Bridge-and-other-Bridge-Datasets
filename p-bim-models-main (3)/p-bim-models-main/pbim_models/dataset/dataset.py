import copy
import numbers
import shutil
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from pbim_models.dataset.anomaly import AnomalyScenario
from pbim_models.dataset.index import WindowIndex
from pbim_models.dataset.model import DatasetMetadata
from pbim_models.dataset.transform import (
    Transform,
    Transforms,
)
from pbim_models.models.util import Stage


def _merge(data: List[Any]):
    if isinstance(data[0], torch.Tensor):
        return torch.stack(data)
    elif isinstance(data[0], numbers.Number | bool):
        return torch.tensor(data)
    elif isinstance(data[0], dict):
        merged = {
            key: _merge([d[key] for d in data])
            for key in data[0]
            if not key.startswith("_")
        }
        singleton = {key: data[0][key] for key in data[0] if key.startswith("_")}
        return {**merged, **singleton}
    else:
        return data


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        offset=0,
        limit: Optional[int] = None,
        transform: Transform = Transforms.identity_transform,
        lazy_init: bool = False,
        in_memory: bool = False,
    ):
        self._dataset_path = dataset_path
        self._offset = offset
        self._transform = transform
        self._f = None
        self._data = None
        # If memory issues, see https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        self._in_memory = in_memory
        self._metadata = self._read_metadata_file()
        self._limit = (
            limit if limit is not None else self._metadata.length - self._offset
        )
        if self._offset >= self._metadata.length:
            raise ValueError(
                f"Offset is greater or equal to the dataset length ({self._offset} >= {self._metadata.length}"
            )
        if self._limit > self._metadata.length - self._offset:
            raise ValueError(
                f"Limit is greater than the dataset length ({self._limit} > {self._metadata.length - self._offset}"
            )
        if not lazy_init:
            self.setup()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def setup(self):
        if not self._in_memory:
            self._f = self._open_dataset()
        else:
            with self._open_dataset() as f:
                self._data = f.read()

    def _open_dataset(self):
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset {self._dataset_path} does not exist")
        return open(self._dataset_path, "rb")

    def _read_metadata_file(self):
        metadata_path = (
            self._dataset_path.parent / f"{self._dataset_path.stem}.metadata.json"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            return DatasetMetadata.from_json(f.read())

    def _parse_timestep(self, data: bytes):
        time_byte_format = "q" if self._metadata.time_byte_size == 8 else "i"
        return struct.unpack(
            f"<{time_byte_format}{len(self._metadata.channel_order) - 1}f", data
        )

    def _read_measurement(self, index: int):
        offset = index * self._metadata.measurement_size_in_bytes
        if self._in_memory:
            data = self._data[
                offset : offset + self._metadata.measurement_size_in_bytes
            ]
        else:
            self._f.seek(offset)
            data = self._f.read(self._metadata.measurement_size_in_bytes)
        values = self._parse_timestep(data)
        return {
            "data": {
                channel: value
                for channel, value in zip(self._metadata.channel_order, values)
            },
        }

    def __len__(self):
        return self._limit

    def __getitem__(self, index: int):
        if not self._f and not self._data:
            self.setup()
        data = self._read_measurement(index)
        return self._transform(self._metadata, data, index)


class WindowedDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        window_size: int,
        horizon_size: int,
        offset=0,
        limit: Optional[int] = None,
        base_transform: Transform = Transforms.identity_transform,
        transform: Transform = Transforms.identity_transform,
        lazy_init: bool = False,
        mirror: bool = False,
        in_memory: bool = False,
    ):
        self._dataset = TimeSeriesDataset(
            dataset_path,
            offset=offset,
            limit=limit,
            transform=base_transform,
            lazy_init=lazy_init,
            in_memory=in_memory,
        )
        self._dataset_path = dataset_path
        self._mirror = mirror
        self._window_size = window_size
        self._horizon_size = 0 if mirror else horizon_size
        self._transform = transform
        self._index = self._build_index(offset, limit)

    def _build_index(self, offset: int, limit: Optional[int]):
        index_path = self._dataset_path.parent / f"{self._dataset_path.stem}.index.json"
        if index_path.exists():
            return WindowIndex(
                index_path,
                self._window_size,
                self._horizon_size,
                offset,
                limit,
            )
        return None

    @staticmethod
    def _copy_sample(sample: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not sample:
            return None
        return {
            key: copy.deepcopy(value)
            if not isinstance(value, torch.Tensor)
            else value.detach().clone()
            for key, value in sample.items()
        }

    def setup(self):
        self._dataset.setup()

    def __len__(self):
        if self._index is not None:
            return len(self._index)
        return len(self._dataset) - self._window_size - self._horizon_size + 1

    def _compute_shifted_index(self, index: int, is_target: bool) -> int:
        if self._index is None:
            return index
        return self._index.compute_shifted_index(index, is_target)

    def _get_window(self, index: int, size: int, is_target: bool = False):
        shifted_index = self._compute_shifted_index(index, is_target)
        window = _merge(
            [self._dataset[i] for i in range(shifted_index, shifted_index + size)]
        )
        if self._index is not None:
            window["anomaly"] = float(self._index.is_anomalous(index, is_target))
        return window

    def __getitem__(self, index: int):
        train = (
            self._transform(
                self._dataset.metadata,
                self._get_window(index, self._window_size, is_target=False),
                index,
            )
            if self._window_size > 0
            else None
        )
        if self._mirror:
            target = self._copy_sample(train) if self._mirror else None
        else:
            target = (
                self._transform(
                    self._dataset.metadata,
                    self._get_window(
                        index + self._window_size, self._horizon_size, is_target=True
                    ),
                    index,
                )
                if self._horizon_size > 0
                else None
            )
        return {
            "train": train,
            "target": target,
            "_metadata": self._dataset.metadata,
        }


class WindowedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any],
        anomaly_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._config = config
        self._anomaly_config = anomaly_config
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
        self._predict_dataloader = None

    def _load_metadata(self, stage: Stage):
        path = Path(
            self._config["dataset"][stage].get(
                "path", self._config["dataset"].get("base", {}).get("path")
            )
        )
        metadata_path = path.parent / f"{path.stem}.metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
        with open(metadata_path, "r") as f:
            return DatasetMetadata.from_json(f.read())

    def _get(self, key: str, stage: Stage, allow_none=False, default=None):
        stage_config = self._config["dataset"].get(stage, {})
        value = stage_config.get(key, self._config["dataset"]["base"].get(key, default))
        if value is None and not allow_none:
            raise ValueError(f"Missing value for key '{key}' in stage '{stage}'")
        return value

    @property
    def scenario_name(self):
        scenario = self._prepare_anomaly_scenario()
        return scenario.name if scenario else "Unnamed"

    def _validate_ratio_usage(self):
        stages_used = [
            s for s in ["train", "val", "test"] if s in self._config["dataset"]
        ]
        ratios_used = any(
            [
                self._get("ratio", stage=s, allow_none=True) is not None
                for s in stages_used
            ]
        )
        if ratios_used:
            stages_with_ratios = [
                s
                for s in stages_used
                if self._get("ratio", stage=s, allow_none=True) is not None
            ]
            # Stages with ratios should use the same dataset path
            if len(set([self._get("path", stage=s) for s in stages_with_ratios])) > 1:
                print(
                    "WARNING: Using different datasets for different stages while using ratios."
                )
                # Stages with ratios should use the same dataset size
                lengths = set(
                    [self._load_metadata(s).length for s in stages_with_ratios]
                )
                if len(lengths) > 1:
                    raise ValueError(
                        "Different dataset sizes for different stages while using ratios."
                    )

            # Ratios have to sum up to something <=1
            ratios = [self._get("ratio", stage=s) for s in stages_with_ratios]
            offsets = [
                self._get("offset", stage=s, default=0) for s in stages_with_ratios
            ]
            if sum(ratios + offsets) > 1:
                raise ValueError("Ratios have to sum up to at most 1.")

        if not ratios_used:
            if not all(
                [
                    self._get("path", stage=s, allow_none=True) is not None
                    for s in stages_used
                ]
            ):
                raise ValueError(
                    f"When not using a ratio split, the dataset path must be set for all stages."
                )

        return ratios_used

    def _compute_ratios(self, stage: Stage, length: int) -> Tuple[int, int | None]:
        ratios_used = self._validate_ratio_usage()
        stage_ratio = self._get("ratio", stage=stage, allow_none=True)
        stage_offset = self._get("offset", stage=stage, default=0) * length
        if not ratios_used or stage_ratio is None:
            return 0, None
        match stage:
            case "train":
                offset, limit = stage_offset, int(
                    length * self._get("ratio", stage="train")
                )
            case "val":
                if not (
                    train_ratio := self._get("ratio", stage="train", allow_none=True)
                ):
                    offset, limit = stage_offset, int(length * stage_ratio)
                else:
                    train_offset = (
                        self._get("offset", stage="train", default=0) * length
                    )
                    offset, limit = int(
                        length * train_ratio
                    ) + train_offset + stage_offset, int(length * stage_ratio)
            case "test":
                train_ratio = self._get("ratio", stage="train", allow_none=True)
                train_offset = self._get("offset", stage="train", default=0) * length
                val_ratio = self._get("ratio", stage="val", allow_none=True)
                val_offset = self._get("offset", stage="val", default=0) * length
                if not train_ratio and not val_ratio:
                    offset, limit = stage_offset, int(length * stage_ratio)
                elif not train_ratio:
                    offset, limit = int(
                        length * val_ratio
                    ) + val_offset + stage_offset, int(length * stage_ratio)
                elif not val_ratio:
                    offset, limit = int(
                        length * train_ratio
                    ) + train_offset + stage_offset, int(length * stage_ratio)
                else:
                    offset, limit = int(
                        length * (train_ratio + val_ratio)
                    ) + train_offset + val_offset + stage_offset, int(
                        length * stage_ratio
                    )
            case _:
                raise ValueError(f"Unknown stage {stage}")
        return offset, limit

    def _build_dataloader(self, stage: Stage):
        from pbim_models.dataset.util import build_dataloader

        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Invalid stage: {stage}")

        if stage not in self._config["dataset"]:
            return None

        def get(key: str, allow_none=False, default=None):
            return self._get(key, stage=stage, allow_none=allow_none, default=default)

        length = self._load_metadata(stage).length
        offset, limit = self._compute_ratios(stage, length)

        anomaly_scenario = self._prepare_anomaly_scenario()
        if anomaly_scenario is not None:
            anomaly_transform = Transforms.IntroduceAnomalyTransform(
                anomaly_scenario.by_channels(), anomaly_scenario.anomaly_probability
            )
        else:
            anomaly_transform = Transforms.identity_transform

        return build_dataloader(
            path=Path(get("path")),
            batch_size=get("batch_size"),
            window_size=get("window_size"),
            horizon_size=get("horizon_size"),
            workers=get("workers"),
            shuffle=get("shuffle", default=False),
            drop_last=get("drop_last", default=True),
            mirror=get("mirror", default=False),
            lazy_init=True,
            base_transform=Transforms.ComposeTransforms(
                Transforms.FilterTransform(Transforms.default_predicate),
                Transforms.to_tensor_transform,
            ),
            transform=Transforms.ComposeTransforms(
                anomaly_transform,
                Transforms.ZScoreTransform(),
                Transforms.GroupChannelsTransform(
                    get("grouping", default="none"),
                    {"z24": ["TBC"], "pbim": ["Temperature"], "lux": ["Temperature"], "none": []},
                ),
            ),
            offset=offset,
            limit=limit,
            in_memory=get("in_memory", default=False),
        )

    def prepare_data(self) -> None:
        dataset_config = self._config["dataset"]["base"]
        if "cache" not in dataset_config or not dataset_config["cache"]:
            return
        path = Path(dataset_config["path"])
        cache_dir = Path(dataset_config["cache"])
        print(f"Caching dataset to '{cache_dir}'")
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, cache_dir)
        shutil.copy(path.parent / f"{path.stem}.metadata.json", cache_dir)
        dataset_config["path"] = cache_dir / path.name

    def setup(self, stage: Optional[str] = None):
        match stage:
            case "fit" | None:
                self._train_dataloader = self._build_dataloader("train")
                self._val_dataloader = self._build_dataloader("val")
            case "test":
                self._test_dataloader = self._build_dataloader("test")
            case "predict":
                self._predict_dataloader = self._build_dataloader("test")
            case _:
                raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._test_dataloader

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self._predict_dataloader

    def _prepare_anomaly_scenario(self) -> AnomalyScenario | None:
        anomaly_config = self._config.get("anomaly_scenario", self._anomaly_config)
        if not anomaly_config:
            return None

        anomaly_type = anomaly_config.get("type", "direct")
        match anomaly_type:
            case "direct":
                updated_config = {**anomaly_config, "name": "Unnamed"}
                return AnomalyScenario.from_config(updated_config)
            case "scenario":
                scenario_file = anomaly_config["file"]
                return AnomalyScenario.from_file(scenario_file)
            case _:
                raise ValueError(f"Unknown anomaly type '{anomaly_type}'")
