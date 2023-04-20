import datetime
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Generator, Any, TextIO, Tuple, Callable, Dict

from pbim_preprocessor.model import (
    ParsedZ24File,
    Measurement,
    ParsedZ24Channel,
    Z24ChannelHeader,
)
from pbim_preprocessor.utils import LOGGER

Z24_EMS_REGEX = re.compile(r"Z24ems(\d+).zip")
Z24_INNER_ZIP_REGEX = re.compile(r"(\d{2})([A-G])(\d{2}).zip")


class Z24UndamagedParser:
    def __init__(self):
        self._acceleration_parser = Z24AccelerationDataParser()
        self._environmental_data_parser = Z24EnvironmentalDataParser()

    @staticmethod
    def _find_and_sort_ems_files(directory: Path) -> List[Path]:
        files = [f for f in os.listdir(directory) if Z24_EMS_REGEX.match(f)]
        files = sorted(files, key=lambda f: int(Z24_EMS_REGEX.match(f).group(1)))
        return [directory / f for f in files]

    @staticmethod
    def _parser_inner_file_name(name: str) -> datetime.datetime:
        match = Z24_INNER_ZIP_REGEX.match(name)
        if not match:
            raise ValueError(f"Could not parse {name}")
        week = int(match.group(1)) - 1  # 0-indexed
        day = ord(match.group(2)) - ord("A") - 2  # 0-indexed but C is Monday
        hour = int(match.group(3))
        return datetime.datetime(
            year=1997, month=11, day=10, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(weeks=week, days=day, hours=hour)

    @staticmethod
    def _extract_inner_file(
        zip_file: zipfile.ZipFile, inner_file: str, output_path: Path
    ) -> Path:
        zip_file.extract(inner_file, path=output_path)
        return output_path / inner_file

    def _parse_inner_file(self, inner_file_path: Path) -> ParsedZ24File:
        with zipfile.ZipFile(inner_file_path) as inner_zip_file:
            data = {}
            pre_env_measurements = None
            post_env_measurements = None
            files = inner_zip_file.namelist()
            for file in files:
                f = io.TextIOWrapper(
                    inner_zip_file.open(file, "r"), encoding="ISO-8859-1", newline=None
                )
                if file.endswith(".aaa"):
                    if "car" in file:
                        continue
                    channel = file[-6:-4]
                    data[channel] = self._acceleration_parser.parse(f)
                elif file.endswith(".env"):
                    if "PRE" in file:
                        pre_env_measurements = self._environmental_data_parser.parse(f)
                    elif "POS" in file:
                        post_env_measurements = self._environmental_data_parser.parse(f)
                    else:
                        raise ValueError(f"Unknown env file {file}")
                else:
                    raise ValueError(f"Unknown file type {file}")
            if not pre_env_measurements or not post_env_measurements:
                raise ValueError(f"Missing env file in {inner_file_path}")
            return ParsedZ24File(
                pre_measurement_environmental_data=pre_env_measurements,
                post_measurement_environmental_data=post_env_measurements,
                acceleration_data=data,
            )

    def _parse_ems_file(
        self,
        ems_file_path: Path,
        output_path: Path,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Generator[ParsedZ24File, Any, None]:
        with zipfile.ZipFile(ems_file_path) as ems_file:
            inner_files = sorted(ems_file.namelist(), key=self._parser_inner_file_name)
            for inner_file in inner_files:

                inner_file_path = self._extract_inner_file(
                    ems_file, inner_file, output_path
                )
                time = self._parser_inner_file_name(inner_file)
                if time < start_time:
                    LOGGER.info(
                        f"Current file contains data for time {time} < {start_time}. Skipping."
                    )
                    continue
                if time > end_time:
                    LOGGER.info(
                        f"Current file contains data for time {time} > {end_time}. Stopping."
                    )
                    break
                LOGGER.info(f"Parsing {inner_file}")
                yield self._parse_inner_file(inner_file_path)
                inner_file_path.unlink()

    def parse(
        self,
        path: Path,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Generator[ParsedZ24File, Any, None]:
        LOGGER.info(
            f"Parsing Z24 undamaged data from {path} between {start_time} and {end_time}"
        )
        if path.is_dir():
            LOGGER.info("Given path is a directory, searching for EMS files")
            ems_files = self._find_and_sort_ems_files(path)
            LOGGER.info(f"Found {len(ems_files)} EMS files")
        else:
            ems_files = [path]
        with tempfile.TemporaryDirectory() as temp_dir:
            for ems_file_path in ems_files:
                LOGGER.info(f"Parsing {ems_file_path}")
                yield from self._parse_ems_file(
                    ems_file_path, Path(temp_dir), start_time, end_time
                )


class Z24AccelerationDataParser:
    @staticmethod
    def _parse_footer(f: TextIO) -> Tuple[datetime.datetime, datetime.datetime]:
        def _find(prefix: str):
            while not (line := f.readline()).startswith(prefix):
                pass
            return line

        start = datetime.datetime.strptime(
            _find(f"Segment #1").split(":", maxsplit=1)[1].strip(),
            "%a %b %d %H:%M:%S %Y",
        )
        _find("Segment #8")
        stop = datetime.datetime.strptime(
            _find(" Stop").split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        return start, stop

    @staticmethod
    def _annotate_times(
        data: List[float], start: datetime.datetime, end: datetime.datetime
    ) -> List[Measurement]:
        measurements = []
        span = (end - start).total_seconds()
        for i, measurement in enumerate(data):
            time = start + datetime.timedelta(seconds=span * i / len(data))
            measurements.append(
                Measurement(measurement=measurement, time=int(time.timestamp() * 1000))
            )
        return measurements

    def parse(self, f: TextIO) -> ParsedZ24Channel:
        name = str(f.readline().strip())
        num_samples = int(f.readline().strip())
        frequency = float(f.readline().strip())
        data = []
        for i in range(num_samples):
            measurement = float(f.readline().strip())
            data.append(measurement)

        start, stop = self._parse_footer(f)
        return ParsedZ24Channel(
            channel_header=Z24ChannelHeader(
                name=name,
                frequency=frequency,
                num_samples=num_samples,
            ),
            measurements=self._annotate_times(data, start, stop),
        )


class Z24EnvironmentalDataParser:
    @staticmethod
    def _parser_header(f: TextIO) -> List[str]:
        headers = [w for w in f.readline().split()]
        return headers[::2]

    @staticmethod
    def _convert(data: List[Measurement], name: str) -> List[Measurement]:
        def _map(m: Measurement, f: Callable[[float], float]) -> Measurement:
            return Measurement(measurement=f(m.measurement), time=m.time)

        match name:
            case "WS":
                return [_map(d, lambda x: x * 5) for d in data]
            case "WD":
                return [_map(d, lambda x: x * 36) for d in data]
            case "AT":
                return [_map(d, lambda x: x * 100 - 40) for d in data]
            case "R":
                return [_map(d, lambda x: 1 if x >= 3 else 0) for d in data]
            case "H":
                return [_map(d, lambda x: x * 10) for d in data]
            case "TE":
                return [_map(d, lambda x: x * 5) for d in data]
            case _:
                return data

    @staticmethod
    def _parse_footer(f: TextIO) -> Tuple[datetime.datetime, datetime.datetime]:
        def _find(prefix: str):
            while not (line := f.readline()).startswith(prefix):
                pass
            return line

        footer = _find("EnvScan started")
        start = datetime.datetime.strptime(
            footer.split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        footer = _find("stopped")
        end = datetime.datetime.strptime(
            footer.split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        return start, end

    @staticmethod
    def _annotate_times(
        data: List[float], times: Tuple[datetime.datetime, datetime.datetime]
    ) -> List[Measurement]:
        measurements = []
        start, end = times
        for i, measurement in enumerate(data):
            time = start + datetime.timedelta(
                seconds=i / len(data) * (end - start).total_seconds()
            )
            measurements.append(
                Measurement(measurement=measurement, time=int(time.timestamp()))
            )
        return measurements

    def parse(self, f: TextIO) -> Dict[str, ParsedZ24Channel]:
        header = self._parser_header(f)
        data = {}
        for line in f:
            if line.strip() == "":
                break
            line_data = [float(w) for w in line.strip().split()]
            for i in range(len(header)):
                if header[i] not in data:
                    data[header[i]] = []
                data[header[i]].append(line_data[i])

        times = self._parse_footer(f)
        data = {k: self._annotate_times(v, times) for k, v in data.items()}

        return {
            header[i]: ParsedZ24Channel(
                channel_header=Z24ChannelHeader(
                    name=header[i],
                    frequency=None,
                    num_samples=len(data[header[i]]),
                ),
                measurements=self._convert(data[header[i]], header[i]),
            )
            for i in range(len(header))
        }
