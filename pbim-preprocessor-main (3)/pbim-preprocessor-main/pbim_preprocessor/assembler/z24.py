import datetime
import statistics
import zipfile
from pathlib import Path
from statistics import mean
from typing import List, Dict, Generator, Any, Tuple, Literal

import numpy as np
import scipy
from typing.io import IO

from pbim_preprocessor.assembler.util import MergeChannelsConfig
from pbim_preprocessor.model import ParsedZ24File, Measurement, EOF
from pbim_preprocessor.parser.z24 import Z24UndamagedParser
from pbim_preprocessor.sampling import SamplingStrategy
from pbim_preprocessor.utils import LOGGER


class Z24EMSAssembler:
    def __init__(
        self,
        path: Path,
        sampling_strategy: SamplingStrategy,
        resolution: float,
        merge_channels_config: List[MergeChannelsConfig] | None = None,
    ):
        self._path = path
        self._parser = Z24UndamagedParser()
        self._sampling_strategy = sampling_strategy
        self._resolution = resolution
        self._merge_channels_config = merge_channels_config or []

    @staticmethod
    def _make_environmental_data(
        data: ParsedZ24File, channels: List[str] | None
    ) -> Dict[str, float]:
        def _mean(values: List[Measurement]) -> float:
            return mean([x.measurement for x in values])

        pre = data.pre_measurement_environmental_data
        post = data.post_measurement_environmental_data
        return {
            channel: (
                _mean(pre[channel].measurements) + _mean(post[channel].measurements)
            )
            / 2
            for channel in pre.keys()
            if channels is None or channel in channels
        }

    @staticmethod
    def _find_start_time(data: ParsedZ24File) -> int:
        channel = list(data.acceleration_data.keys())[0]
        return min([x.time for x in data.acceleration_data[channel].measurements])

    def _make_acceleration_data(self, data: ParsedZ24File, channels: List[str] | None):
        def _find_until(
            values: List[Measurement], timestamp: float, offset: int = 0
        ) -> List[Measurement]:
            results = []
            for value in values[offset:]:
                if value.time <= timestamp:
                    results.append(value)
                else:
                    break
            return results

        sampled = {}
        for channel, values in data.acceleration_data.items():
            if channels is not None and channel not in channels:
                continue
            # No need to sample
            if self._resolution == 0:
                sampled[channel] = [m.measurement for m in values.measurements]
                continue
            # All in milliseconds
            end_time = (
                min([x.time for x in values.measurements]) + self._resolution * 1000
            )
            offset = 0
            sampled_values = []
            while offset < len(values.measurements):
                sample = _find_until(values.measurements, end_time, offset)
                target_time = datetime.datetime.fromtimestamp(
                    end_time / 1000 - self._resolution / 2
                )
                value = self._sampling_strategy.sample(
                    sample,
                    target_time,
                )
                if value is None:
                    LOGGER.warn(
                        f"Failed to sample value for channel {channel} at time {target_time}."
                    )
                    continue
                end_time += self._resolution * 1000
                sampled_values.append(value)
                offset += len(sample)
            sampled[channel] = sampled_values
        return sampled

    def _make_measurement_dict(
        self,
        data: ParsedZ24File,
        channels: List[str] | None,
    ) -> Generator[Dict[str, float], Any, None]:
        start_time = self._find_start_time(data)
        environmental_data = self._make_environmental_data(data, channels)
        acceleration_data = self._make_acceleration_data(data, channels)
        lengths = list(set([len(x) for x in acceleration_data.values()]))
        if len(lengths) != 1:
            shortest = min(lengths)
            LOGGER.warn(
                f"Unequal lengths for acceleration data: {lengths}. Cutting to {shortest}."
            )
            acceleration_data = {
                channel: value[:shortest]
                for channel, value in acceleration_data.items()
            }
        for i in range(lengths[0]):
            sample_acceleration_data = {
                channel: value[i] for channel, value in acceleration_data.items()
            }
            # Start time + middle of current sample in milliseconds
            time = start_time + 1.5 * i * self._resolution * 1000
            yield self._merge_channels(
                {"time": time, **environmental_data, **sample_acceleration_data}
            )

    def assemble(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: List[str] | None = None,
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        for data in self._parser.parse(self._path, start_time, end_time):
            yield from self._make_measurement_dict(data, channels)
            yield EOF()

    def _merge_channels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for merge_config in self._merge_channels_config:
            if not all([x in data for x in merge_config.channels]):
                raise ValueError(f"Missing channels for merge: {merge_config.channels}")
            values = [data[x] for x in merge_config.channels]
            data[merge_config.name] = statistics.mean(values)
            if merge_config.remove_original:
                for channel in merge_config.channels:
                    del data[channel]
        return data


class Z24PDTAssembler:
    FILE_NAMES = ["pdt_01-08.zip", "pdt_09_17.zip"]
    NUM_SETUPS = 9
    SAMPLING_RATE = 100  # Hz

    def __init__(self, path: Path):
        self._path = path
        pass

    def _make_file_path(self, scenario: int) -> Path:
        assert scenario in range(1, 18)
        return self._path / self.FILE_NAMES[scenario // 9]

    @staticmethod
    def _load_scenario_data(f: IO[bytes]) -> Tuple[np.ndarray, List[str]]:
        mat = scipy.io.loadmat(f)
        data = mat["data"]
        labels = [label.strip() for label in mat["labelshulp"].tolist()]
        return data, labels

    def _combine(
        self, data: List[Tuple[np.ndarray, List[str]]], channels: List[str] | None
    ) -> Generator[Dict[str, float], Any, None]:
        step_data = {}
        length = min([x.shape[0] for (x, _) in data])
        for i in range(length):
            LOGGER.info(f"Processing step {i+1}/{length}.")
            for (measurements, labels) in data:
                for j, label in enumerate(labels):
                    if channels is not None and label not in channels:
                        continue
                    step_data[label] = measurements[i, j]
            step_data["time"] = i / self.SAMPLING_RATE
            yield step_data

    @staticmethod
    def _make_path_in_zip_file(
        scenario: int, scenario_type: Literal["avt", "fvt"]
    ) -> str:
        if scenario == 17 and scenario_type == "fvt":
            # For some reason, the 17th scenario is in a all upper-case folder.
            return "17/FVT/"
        return f"{scenario:02d}/{scenario_type}/"

    def assemble(
        self,
        scenario: int,
        scenario_type: Literal["avt", "fvt"],
        channels: List[str] | None = None,
    ) -> Generator[Dict[str, float], Any, None]:
        with zipfile.ZipFile(self._make_file_path(scenario)) as zip_file:
            data = []
            for setup in range(1, self.NUM_SETUPS + 1):
                setup_path = (
                    self._make_path_in_zip_file(scenario, scenario_type)
                    + f"{scenario:02d}setup{setup:02d}.mat"
                )
                with zip_file.open(setup_path) as setup_file:
                    data.append(self._load_scenario_data(setup_file))
            yield from self._combine(data, channels)
