import csv
import dataclasses
import datetime
import re
import struct
from enum import Enum
from pathlib import Path
from typing import List, Dict, BinaryIO, Tuple

from pbim_preprocessor.model import (
    ParsedPBimChannel,
    Measurement,
    PBimChannelHeader,
    GlobalHeader,
)

MAGIC_FREQUENCY_CONSTANT = 270135
ALL_EXTENSIONS = ["R32", "DAT"]


def ensure_files_exist(directory: Path, name: str, extensions: List[str]) -> None:
    for extension in extensions:
        if not (directory / f"{name}.{extension}").exists():
            raise MeasurementFileMissingError(name, extension)


class MeasurementFileMissingError(Exception):
    def __init__(self, name: str, extension: str):
        self._name = name
        self._extensions = extension


class TimeChannel(Enum):
    STANDARD_1 = "Zeit  1 - Standardmessrate"
    STANDARD_2 = "Zeit  2 - Standardmessrate"
    STANDARD_4 = "Zeit  4 - Standardmessrate"
    FAST_3 = "Zeit  3 - Schnelle Messrate"
    FAST_5 = "Zeit  5 - Schnelle Messrate"
    SLOW_5 = "Zeit  5 - Langsame Messrate"


CHANNEL_TIME_MAP = {
    # Measurement amplifier 1
    "MS1L": TimeChannel.STANDARD_1,
    "MS2L": TimeChannel.STANDARD_1,
    "MS3Q": TimeChannel.STANDARD_1,
    "MS3L": TimeChannel.STANDARD_1,
    "MS4Q": TimeChannel.STANDARD_1,
    "MS4L": TimeChannel.STANDARD_1,
    "MS5L": TimeChannel.STANDARD_1,
    "MS6L": TimeChannel.STANDARD_1,
    # Measurement amplifier 2
    "MS7Q": TimeChannel.STANDARD_2,
    "MS7L": TimeChannel.STANDARD_2,
    "MS8Q": TimeChannel.STANDARD_2,
    "MS8L": TimeChannel.STANDARD_2,
    "MS27v": TimeChannel.STANDARD_2,
    "MS28v": TimeChannel.STANDARD_2,
    "MS29v": TimeChannel.STANDARD_2,
    "MS30h": TimeChannel.STANDARD_2,
    # Measurement amplifier 3
    "MS23": TimeChannel.FAST_3,
    "MS24": TimeChannel.FAST_3,
    "MS25": TimeChannel.FAST_3,
    "MS26": TimeChannel.FAST_3,
    # Measurement amplifier 4
    "MS9Q": TimeChannel.STANDARD_4,
    "MS10Q": TimeChannel.STANDARD_4,
    "MS11Q": TimeChannel.STANDARD_4,
    "MS12Q": TimeChannel.STANDARD_4,
    "MS13Q": TimeChannel.STANDARD_4,
    "MS14Q": TimeChannel.STANDARD_4,
    "MS15Q": TimeChannel.STANDARD_4,
    "MS16Q": TimeChannel.STANDARD_4,
    # Measurement amplifier 5
    "MS17": TimeChannel.FAST_5,
    "MS18": TimeChannel.FAST_5,
    "MS19": TimeChannel.FAST_5,
    "MS20": TimeChannel.FAST_5,
    "MS21": TimeChannel.FAST_5,
    "MS22": TimeChannel.FAST_5,
    "MS6T1u": TimeChannel.SLOW_5,
    "MS6T2o": TimeChannel.SLOW_5,
}
POST_PROCESSABLE_CHANNELS = list(CHANNEL_TIME_MAP.keys())
TIME_CHANNELS = [channel.value for channel in TimeChannel]


class PBimRealDataParser:
    def __init__(self):
        self._metadata_parser = PBimMetadataParser()
        self._data_parser = PBimDataParser()

    def parse(
        self, directory: Path, name: str, channels: List[str]
    ) -> Dict[str, ParsedPBimChannel]:
        ensure_files_exist(directory, name, ALL_EXTENSIONS)
        channels_with_time = channels + [channel.value for channel in TimeChannel]
        global_header, channel_header = self._metadata_parser.parse(directory, name)
        start_date, start_time = global_header.date, global_header.time
        t0 = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=start_time.hour,
            minute=start_time.minute,
            second=start_time.second,
            microsecond=start_time.microsecond,
        )
        filtered_headers = [
            header for header in channel_header if header.name in channels_with_time
        ]
        data = self._data_parser.parse_all(directory, name, filtered_headers)
        time_channels = {
            name: parsed_channel.measurements
            for name, parsed_channel in data.items()
            if name in TIME_CHANNELS
        }
        for name, parsed_channel in data.items():
            if name in POST_PROCESSABLE_CHANNELS and name in channels:
                self._post_process(parsed_channel, time_channels, t0)
        return {
            name: parsed_channel
            for name, parsed_channel in data.items()
            if name in channels
        }

    def reset(self):
        self._metadata_parser.reset()

    @staticmethod
    def _post_process(
        channel: ParsedPBimChannel,
        time_channels: Dict[str, List[Measurement]],
        start_time: datetime.datetime,
    ) -> None:
        time_channel = CHANNEL_TIME_MAP[channel.channel_header.name]
        if not time_channel:
            return
        time_data = time_channels[time_channel.value]
        assert len(time_data) == len(channel.measurements)
        start_time_in_milliseconds = int(start_time.timestamp() * 1000)
        for i, dat in enumerate(channel.measurements):
            # Round to the nearest millisecond
            dat.time = start_time_in_milliseconds + int(time_data[i].measurement * 1000)


class PBimDataParser:
    def parse(
        self, directory: Path, name: str, channel: PBimChannelHeader
    ) -> ParsedPBimChannel:
        encoding = self._infer_encoding(channel)
        actual_offset = self._compute_actual_offset(channel)
        with open(directory / f"{name}.R32", "rb") as f:
            f.seek(actual_offset)
            return ParsedPBimChannel(
                channel_header=channel,
                measurements=[
                    self._parse_measurement(f, encoding)
                    for _ in range(channel.measurements)
                ],
            )

    def parse_all(
        self, directory: Path, name: str, channel_headers: List[PBimChannelHeader]
    ) -> Dict[str, ParsedPBimChannel]:
        return {
            channel_header.name: self.parse(directory, name, channel_header)
            for channel_header in channel_headers
        }

    def _compute_actual_offset(self, channel: PBimChannelHeader) -> int:
        encoding = self._infer_encoding(channel)
        size = struct.calcsize(encoding)
        # offset is given in "records" instead of bytes initially
        return (channel.data_offset - 1) * size

    @staticmethod
    def _parse_measurement(f: BinaryIO, encoding: str) -> Measurement:
        # data is encoded in little-endian
        data = f.read(struct.calcsize(encoding))
        value = struct.unpack(encoding, data)[0]
        return Measurement(measurement=value)

    @staticmethod
    def _infer_encoding(channel: PBimChannelHeader) -> str:
        match channel.dtype.lower():
            case "int32":
                return "<i"
            case "int16":
                return "<H"
            case "real32":
                return "<f"
            case "real64":
                return "<d"
            case _:
                raise ValueError(f"Unknown data type {channel.dtype}")


HEADER_MAGIC = "DIAEXTENDED"
ENCODING = "cp1252"


class PBimMetadataParserState(Enum):
    NONE = 0
    GLOBAL_HEADER = 1
    CHANNEL_HEADER = 2


class MetadataParserError(Exception):
    def __init__(self, state: PBimMetadataParserState, line: str, line_number: int):
        self._state = state
        self._line = line
        self.line = line_number

    def __repr__(self):
        return f"ParserError(state={self._state}, line={self._line}, line_number={self.line})"


class PBimMetadataParser:
    def __init__(self):
        self._state = PBimMetadataParserState.NONE
        self._position = 0
        self._current_global_header = None
        self._current_channel_header = None
        self._channel_headers = []

    def parse(
        self, directory: Path, name: str
    ) -> Tuple[GlobalHeader, List[PBimChannelHeader]]:
        with open(directory / f"{name}.DAT", "r", encoding=ENCODING) as f:
            for line in f.readlines():
                self._parse_line(line.strip())
        return self._current_global_header, self._channel_headers

    def reset(self):
        self._state = PBimMetadataParserState.NONE
        self._position = 0
        self._current_global_header = None
        self._current_channel_header = None
        self._channel_headers = []

    def _parse_line(self, line: str):
        if line == HEADER_MAGIC:
            return
        elif line.startswith("#"):
            self._parse_command(line)
        elif re.match("^\\d+,.*$", line):
            self._parse_data(line)
        else:
            print(line)
            raise MetadataParserError(
                state=self._state, line=line, line_number=self._position
            )
        self._position += 1

    def _parse_command(self, line: str):
        match line:
            case "#BEGINGLOBALHEADER":
                if self._current_global_header:
                    raise MetadataParserError(
                        state=self._state, line=line, line_number=self._position
                    )
                self._current_global_header = GlobalHeader()
                self._state = PBimMetadataParserState.GLOBAL_HEADER
            case "#ENDGLOBALHEADER":
                self._state = PBimMetadataParserState.NONE
            case "#BEGINCHANNELHEADER":
                self._state = PBimMetadataParserState.CHANNEL_HEADER
                self._current_channel_header = PBimChannelHeader()
            case "#ENDCHANNELHEADER":
                if self._current_channel_header:
                    self._channel_headers.append(self._current_channel_header)
                    self._current_channel_header = None
                self._state = PBimMetadataParserState.NONE
            case _:
                raise MetadataParserError(
                    state=self._state, line=line, line_number=self._position
                )

    def _parse_data(self, line: str):
        match self._state:
            case PBimMetadataParserState.GLOBAL_HEADER:
                self._parse_global_header_line(line)
            case PBimMetadataParserState.CHANNEL_HEADER:
                self._parse_channel_header_line(line)
            case _:
                raise MetadataParserError(
                    state=self._state, line=line, line_number=self._position
                )

    def _parse_global_header_line(self, line: str):
        command, value = line.split(",", maxsplit=1)
        match int(command):
            case 104:
                self._current_global_header = dataclasses.replace(
                    self._current_global_header,
                    date=datetime.datetime.strptime(value, "%m-%d-%Y"),
                )
            case 105:
                self._current_global_header = dataclasses.replace(
                    self._current_global_header,
                    time=datetime.datetime.strptime(value, "%H:%M:%S"),
                )
            case _:
                return

    def _parse_channel_header_line(self, line: str):
        command, value = line.split(",", maxsplit=1)
        match int(command):
            case 200:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, name=value
                )
            case 202:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, unit=value
                )
            case 210:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, channel_type=value
                )
            case 213:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, encoding=value
                )
            case 214:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, dtype=value
                )
            case 215:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, bit_mask=int(value)
                )
            case 220:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, measurements=int(value)
                )
            case 221:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, data_offset=int(value)
                )
            case _:
                return


class PBimArtificialDataParser:
    def __init__(self, delimiter: str = ";"):
        self._delimiter = delimiter

    def _parse_line(self, line: str) -> List[str]:
        return [entry.strip() for entry in line.strip().split(self._delimiter)[:-1]]

    def parse(
        self, input_file: Path, channels: List[str] | None = None
    ) -> Dict[str, List[Measurement]]:
        with open(input_file, "r") as f:
            header = self._parse_line(f.readline())
            actual_channels = [
                channel for channel in header if channels is None or channel in channels
            ]
            parsed_data = {channel: [] for channel in actual_channels}
            for line in f.readlines():
                data = [
                    float(value.replace(",", ".")) for value in self._parse_line(line)
                ]
                time = int(data[0])
                for channel, value in zip(header, data):
                    if channel in actual_channels:
                        parsed_data[channel].append(
                            Measurement(measurement=value, time=time)
                        )
            return parsed_data
