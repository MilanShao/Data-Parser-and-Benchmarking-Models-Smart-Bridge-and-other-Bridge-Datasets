import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ChannelStatistics:
    mean: float
    var: float
    std: float
    min: float
    max: float


@dataclass_json
@dataclass
class DatasetMetadata:
    channel_order: List[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    measurement_size_in_bytes: int
    resolution: Optional[int]
    length: int
    statistics: Dict[str, ChannelStatistics]
    time_byte_size: int


@dataclass_json
@dataclass
class CutIndexEntry:
    start_measurement_index: int
    end_measurement_index: int
    anomalous: bool
