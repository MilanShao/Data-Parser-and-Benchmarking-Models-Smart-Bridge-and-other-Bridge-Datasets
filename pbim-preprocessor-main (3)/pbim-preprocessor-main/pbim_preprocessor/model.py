from dataclasses import dataclass
from typing import Optional, List, Dict
import datetime

from dataclasses_json import dataclass_json


@dataclass(frozen=True)
class GlobalHeader:
    date: Optional[datetime.datetime] = None
    time: Optional[datetime.datetime] = None


@dataclass_json
@dataclass(frozen=True)
class PBimChannelHeader:
    name: Optional[str] = None
    unit: Optional[str] = None
    encoding: Optional[str] = None
    dtype: Optional[str] = None
    bit_mask: Optional[int] = None
    channel_type: Optional[str] = None
    data_offset: Optional[int] = None
    measurements: Optional[int] = None


@dataclass
class Measurement:
    measurement: float
    # Epoch in milliseconds
    time: Optional[int] = None


@dataclass_json
@dataclass
class ParsedPBimChannel:
    channel_header: PBimChannelHeader
    measurements: List[Measurement]


@dataclass_json
@dataclass
class Z24ChannelHeader:
    name: str
    frequency: Optional[float]
    num_samples: int


@dataclass_json
@dataclass
class ParsedZ24Channel:
    channel_header: Z24ChannelHeader
    measurements: List[Measurement]


@dataclass_json
@dataclass
class ParsedZ24File:
    acceleration_data: Dict[str, ParsedZ24Channel]
    pre_measurement_environmental_data: Dict[str, ParsedZ24Channel]
    post_measurement_environmental_data: Dict[str, ParsedZ24Channel]


class EOF:
    pass


@dataclass
class ParsedLuxFile:
    channels: List[str]
