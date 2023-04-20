import abc
import calendar
import datetime
import math
from typing import List, Tuple

import numpy as np

from pbim_preprocessor.utils import LOGGER


def _find_group_index(index: int, start_and_end_indices: List[Tuple[int, int]]):
    for i, (start, end) in enumerate(start_and_end_indices):
        if start <= index < end:
            return i

    end_start_pairs = [(0, start_and_end_indices[0][0])]
    for i in range(len(start_and_end_indices) - 1):
        first_start, first_end = start_and_end_indices[i]
        second_start, second_end = start_and_end_indices[i + 1]
        end_start_pairs.append((first_end, second_start))

    for i, (end, start) in enumerate(end_start_pairs):
        if end <= index < start:
            return i

    return None


def _merge_windows(sample_indices: List[int], window_size: int):
    last_index = -math.inf
    final_indices = []
    for index in sample_indices:
        if index < last_index + window_size:
            adjusted_index = last_index + window_size
            remaining_length = index - last_index
        else:
            adjusted_index = index
            remaining_length = window_size
        last_index = index
        final_indices.extend([adjusted_index + i for i in range(remaining_length)])
    return final_indices


def _sample_next_interval(
    time: np.ndarray,
    sample_length_in_seconds: float,
    start_and_end_indices: List[Tuple[int, int]],
    start_index: int,
):
    current_length = 0

    if (group_index := _find_group_index(start_index, start_and_end_indices)) is None:
        return None

    sample_indices = []
    current_start, current_end = start_and_end_indices[group_index]
    if start_index < current_start:
        start_index = current_start
    current_index = start_index
    while current_length < sample_length_in_seconds:
        sample_indices.append(current_index)
        current_length = (time[current_index] - time[start_index]) / 1000
        current_index += 1
        if current_index >= current_end:
            group_index += 1
            if group_index >= len(start_and_end_indices):
                break
            current_start, current_end = start_and_end_indices[group_index]
            current_index = current_start

    return sample_indices, current_index


class DatasetSamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        pass


class UniformSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        samples_per_group = [
            (end - start) * self._num_samples / available_samples
            for start, end in start_and_end_indices
        ]
        sample_indices = []
        for (start, end), desired_samples in zip(
            start_and_end_indices, samples_per_group
        ):
            windows_per_group = max(round(desired_samples) - self._window_size + 1, 1)
            step = round((end - start - self._window_size) / windows_per_group)
            sample_indices.extend([start + i * step for i in range(windows_per_group)])
        return _merge_windows(sample_indices, self._window_size)


class RandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        window_indices = []
        for start, end in start_and_end_indices:
            window_indices.extend(list(range(start, end - self._window_size)))
        sampled_indices = np.random.choice(
            window_indices, self._num_samples, replace=False
        ).tolist()
        return _merge_windows(sorted(sampled_indices), self._window_size)


class WeightedRandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    @staticmethod
    def _compute_weight(timestamp: int) -> float:
        time = datetime.datetime.utcfromtimestamp(timestamp / 1000)
        if 6 <= time.hour <= 9:
            return 1
        if 16 <= time.hour <= 19:
            return 1
        if 20 <= time.hour <= 5:
            return 0.25
        return 0.5

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        window_indices = []
        for start, end in start_and_end_indices:
            window_indices.extend(list(range(start, end - self._window_size)))
        weights = np.array(
            [self._compute_weight(time[index]) for index in window_indices]
        )
        weights = weights / np.sum(weights)
        sampled_indices = np.random.choice(
            window_indices,
            self._num_samples // self._window_size,
            replace=False,
            p=weights,
        ).tolist()
        final_indices = []
        last_index = -math.inf
        for index in sorted(sampled_indices):
            if index < last_index + self._window_size:
                adjusted_index = last_index + self._window_size
                remaining_length = index - last_index
            else:
                adjusted_index = index
                remaining_length = self._window_size
            last_index = index
            final_indices.extend([adjusted_index + i for i in range(remaining_length)])
        return final_indices


class IntervalSamplingStrategy(DatasetSamplingStrategy):
    def __init__(
        self,
        interval_length_in_seconds: int,
        samples_per_interval: int,
        sample_length_in_seconds: int,
    ):
        self._interval_length_in_seconds = interval_length_in_seconds
        self._samples_per_interval = samples_per_interval
        self._sample_length_in_seconds = sample_length_in_seconds

    @staticmethod
    def _make_datetime(timestamp: float) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(timestamp / 1000)

    @staticmethod
    def _make_timestamp(dt: datetime.datetime) -> float:
        return calendar.timegm(dt.timetuple()) * 1000

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        final_indices = []
        start_date = self._make_datetime(time[0])
        end_date = self._make_datetime(time[-1])
        current_date = start_date
        while current_date < end_date:
            for i in range(self._samples_per_interval):
                interval_start = current_date + datetime.timedelta(
                    seconds=i
                    * self._interval_length_in_seconds
                    / self._samples_per_interval
                )
                if interval_start > end_date:
                    break
                if len(final_indices) > 0 and interval_start < self._make_datetime(
                    time[final_indices[-1]]
                ):
                    LOGGER.warn(
                        f"Skipping interval start {interval_start} because it is before the last sample."
                    )
                    continue
                start_index = np.searchsorted(
                    time, self._make_timestamp(interval_start)
                ).item()
                interval = _sample_next_interval(
                    time,
                    self._sample_length_in_seconds,
                    start_and_end_indices,
                    start_index,
                )
                if not interval:
                    break
                sample_indices, end_index = interval
                final_indices.extend(sample_indices)
            current_date += datetime.timedelta(seconds=self._interval_length_in_seconds)
        return final_indices


class HourlySamplingStrategy(IntervalSamplingStrategy):
    def __init__(self, samples_per_hour: int, sample_length_in_seconds: int):
        super().__init__(3600, samples_per_hour, sample_length_in_seconds)


class MinutelySamplingStrategy(IntervalSamplingStrategy):
    def __init__(self, samples_per_minute: int, sample_length_in_seconds: int):
        super().__init__(60, samples_per_minute, sample_length_in_seconds)
