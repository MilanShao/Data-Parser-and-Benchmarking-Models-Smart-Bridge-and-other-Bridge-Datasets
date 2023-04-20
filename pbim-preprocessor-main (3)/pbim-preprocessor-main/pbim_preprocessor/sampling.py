import abc
import datetime
from statistics import mean
from typing import List, Optional

from pbim_preprocessor.model import Measurement


class SamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def sample(
        self, data: List[Measurement], target: datetime.datetime
    ) -> Optional[float]:
        pass


class MeanSamplingStrategy(SamplingStrategy):
    def sample(
        self, data: List[Measurement], target: datetime.datetime
    ) -> Optional[float]:
        return mean([m.measurement for m in data]) if len(data) > 0 else None


class LinearInterpolationSamplingStrategy(SamplingStrategy):
    def sample(
        self, data: List[Measurement], target: datetime.datetime
    ) -> Optional[float]:
        if len(data) == 0:
            return None
        timestamp = int(target.timestamp()) * 1000
        left = self._find_left(data, timestamp)
        right = self._find_right(data, timestamp)
        if left.time == right.time:
            return left.measurement
        t = (timestamp - left.time) / (right.time - left.time)
        return left.measurement + t * (right.measurement - left.measurement)

    @staticmethod
    def _find_left(data: List[Measurement], target_timestamp: int) -> Measurement:
        current_index = 0
        while current_index < len(data) and data[current_index].time < target_timestamp:
            current_index += 1
        return data[max(0, current_index - 1)]

    @staticmethod
    def _find_right(data: List[Measurement], target_timestamp: int) -> Measurement:
        current_index = len(data) - 1
        while current_index > 0 and data[current_index].time > target_timestamp:
            current_index -= 1
        return data[min(len(data) - 1, current_index + 1)]
