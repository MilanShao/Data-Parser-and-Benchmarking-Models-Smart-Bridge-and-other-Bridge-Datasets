from dataclasses import dataclass
from typing import Dict

import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ChannelStatistics:
    mean: float
    var: float
    std: float
    min: float
    max: float


class StatisticsCollector:
    def __init__(self):
        self._stats = {}

    def add(self, key: str, value: float):
        if np.isnan(value):
            print(f"[WARNING] Encountered invalid value '{value}' for key {key}.")
        if key not in self._stats:
            self._stats[key] = []
        self._stats[key].append(value)

    def add_all(self, data: Dict[str, float]):
        for key, value in data.items():
            self.add(key, value)

    def get_mean(self, key: str) -> float:
        return np.mean(self._stats[key]).item()

    def get_std(self, key: str) -> float:
        return np.std(self._stats[key]).item()

    def get_min(self, key: str) -> float:
        return np.min(self._stats[key]).item()

    def get_max(self, key: str) -> float:
        return np.max(self._stats[key]).item()

    def get_variance(self, key: str) -> float:
        return np.var(self._stats[key]).item()

    def get_channel_statistics(self, key: str) -> ChannelStatistics:
        return ChannelStatistics(
            mean=self.get_mean(key),
            var=self.get_variance(key),
            std=self.get_std(key),
            min=self.get_min(key),
            max=self.get_max(key),
        )

    def get_all_channel_statistics(self) -> Dict[str, ChannelStatistics]:
        return {key: self.get_channel_statistics(key) for key in self._stats}
