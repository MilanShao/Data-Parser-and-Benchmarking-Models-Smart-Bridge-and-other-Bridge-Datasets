import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from pbim_models.dataset.model import CutIndexEntry


class WindowIndex:
    def __init__(
        self,
        index_file_path: Path,
        window_size: int,
        horizon_size: int,
        offset: int = 0,
        limit: Optional[int] = None,
    ):
        self._index = self._load_index(index_file_path)
        self._cumulative_index = self._compute_cumulative_index()
        self._window_size = window_size
        self._horizon_size = horizon_size
        self._offset = offset
        self._limit = limit
        # Index i contains number of windows in cut i.
        self._window_index = self._compute_window_index()
        # Index i contains number of windows up to cut i.
        self._cumulative_window_index = self._compute_cumulative_window_index()
        self._max_index = self._cumulative_window_index[-1]

    def __len__(self):
        return self._max_index

    @staticmethod
    def _apply_lower_bound(
        measurements_per_cut: np.ndarray, lower_bound: int
    ) -> np.ndarray:
        cumulative_measurements_per_cut = np.cumsum(measurements_per_cut)
        indices_to_skip = np.argmax(cumulative_measurements_per_cut > lower_bound)
        samples_to_skip_from_first_cut = (
            lower_bound - cumulative_measurements_per_cut[indices_to_skip - 1]
            if indices_to_skip > 0
            else lower_bound
        )
        measurements_per_cut = measurements_per_cut[indices_to_skip:]
        measurements_per_cut[0] -= samples_to_skip_from_first_cut
        return measurements_per_cut

    def _compute_window_index(self):
        measurements_per_cut = np.array(
            [
                entry.end_measurement_index - entry.start_measurement_index
                for entry in self._index
            ]
        )
        if self._offset > 0:
            cumulative_measurements_per_cut = np.cumsum(measurements_per_cut)
            indices_to_skip = np.argmax(cumulative_measurements_per_cut > self._offset)
            samples_to_skip_from_first_cut = (
                self._offset - cumulative_measurements_per_cut[indices_to_skip - 1]
                if indices_to_skip > 0
                else self._offset
            )
            measurements_per_cut = measurements_per_cut[indices_to_skip:]
            measurements_per_cut[0] -= samples_to_skip_from_first_cut
        if self._limit is not None:
            np.flip(measurements_per_cut)
            cumulative_measurements_per_cut = np.cumsum(measurements_per_cut)
            indices_to_keep = np.argmax(cumulative_measurements_per_cut > self._limit)
            measurements_per_cut = measurements_per_cut[: indices_to_keep + 1]
            samples_to_skip_from_first_cut = (
                cumulative_measurements_per_cut[indices_to_keep] - self._limit
            )
            measurements_per_cut[0] -= samples_to_skip_from_first_cut
            np.flip(measurements_per_cut)

        return measurements_per_cut - self._window_size - self._horizon_size + 1

    def _compute_cumulative_window_index(self):
        return np.cumsum(self._compute_window_index())

    @staticmethod
    def _load_index(index_file_path: Path):
        with index_file_path.open("r") as f:
            data = json.load(f)
        return [CutIndexEntry.from_dict(entry) for entry in data]

    def _compute_windows_up_to_current_cut(self, index: int) -> Tuple[int, int]:
        windows = 0
        i = 0
        for i, current_cut_windows in enumerate(self._window_index):
            if windows + current_cut_windows > index:
                break
            windows += current_cut_windows
        return windows, i

    def compute_shifted_index(self, index: int, is_target: bool) -> int:
        if is_target:
            index -= self._horizon_size
        assert index < self._max_index
        windows_up_to_current_cut, cut_index = self._compute_windows_up_to_current_cut(
            index
        )
        relative_index = index - windows_up_to_current_cut
        measurements_to_skip = self._cumulative_index[cut_index]
        return measurements_to_skip + relative_index

    def is_anomalous(self, index: int, is_target: bool) -> bool:
        if is_target:
            index -= self._horizon_size
        assert index < self._max_index
        windows_up_to_current_cut, cut_index = self._compute_windows_up_to_current_cut(
            index
        )
        return self._index[cut_index].anomalous

    def _compute_cumulative_index(self):
        return np.cumsum(
            [0]
            + [
                entry.end_measurement_index - entry.start_measurement_index
                for entry in self._index
            ]
        )
