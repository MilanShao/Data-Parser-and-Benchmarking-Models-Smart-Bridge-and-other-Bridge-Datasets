import datetime
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path
from typing import List, Dict, BinaryIO, Literal

from pbim_preprocessor.model import ParsedPBimChannel, Measurement
from pbim_preprocessor.parser.pbim import PBimRealDataParser, PBimArtificialDataParser
from pbim_preprocessor.processor.processor import Processor
from pbim_preprocessor.utils import LOGGER


class PBimRealDataProcessor(Processor):

    FILE_NAME_PATTERN = re.compile(r"^Job1_(\d{4})_(\d{2})_(\d{2})_.*$")

    def __init__(
        self,
        zip_file_path: Path,
        output_base_path: Path,
        tmp_path: Path,
        file_names: List[str],
        channels: List[str],
    ):
        self._zip_file_path = zip_file_path
        self._output_base_path = output_base_path
        self._tmp_path = tmp_path
        self._names = file_names
        self._channels = channels
        self._parser = PBimRealDataParser()
        self._identifier = os.getpid()

    def process(self):
        total = len(self._names)
        for i, raw_name in enumerate(self._names):
            LOGGER.info(
                f"Started processing '{raw_name}' ({i + 1}/{total})", self._identifier
            )
            self._parser.reset()
            path = self._unpack(raw_name)
            name = Path(raw_name).stem
            LOGGER.info(f"Unpacked '{name}' ({i + 1}/{total})", self._identifier)
            data = self._parser.parse(path, name, self._channels)
            LOGGER.info(f"Parsed '{name}' ({i + 1}/{total})", self._identifier)
            self.serialize(data, name)
            LOGGER.info(f"Saved '{name}' ({i + 1}/{total})", self._identifier)
            shutil.rmtree(path)
            LOGGER.info(
                f"Finished processing '{name}' ({i + 1}/{total})", self._identifier
            )

    def _unpack(self, name: str) -> Path:
        tmp_path = self._tmp_path / name
        tmp_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self._zip_file_path, "r") as zip_file:
            for extension in ["DAT", "R32"]:
                data = zip_file.read(f"{name}.{extension}")
                out_file_path = tmp_path / f"{Path(name).stem}.{extension}"
                out_file_path.write_bytes(data)
        return tmp_path

    def serialize(self, data: Dict[str, ParsedPBimChannel], name: str):
        for channel_name, parsed_channel in data.items():
            path = self.get_output_path(name, channel_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = b"".join(
                [self._serialize_measurement(m) for m in parsed_channel.measurements]
            )
            with open(path, "a+b") as f:
                f.write(serialized)

    @staticmethod
    def _serialize_measurement(measurement: Measurement) -> bytes:
        return struct.pack("<qf", measurement.time, measurement.measurement)

    def get_output_path(self, file_name: str, channel_name: str) -> Path:
        match = self.FILE_NAME_PATTERN.match(file_name)
        if not match:
            raise ValueError(
                f"File name '{file_name}' does not match the expected pattern"
            )
        year, month, day = match.groups()
        return (
            self._output_base_path
            / f"{year}"
            / f"{month}"
            / f"{day}"
            / f"{channel_name}.dat"
        )


PBimArtificialScenario = Literal["N", "S0", "S1", "S2", "S3"]


class PBimArtificialDataProcessor(Processor):

    NAME_REGEX = re.compile(r"^MessSimu_MQ_(\d)_(N|S0|S1|S2|S3)_(\d+)_All_V.csv$")
    BASE_TIME = datetime.datetime(2018, 1, 1, 0, 0, 0)
    CHANNELS = [
        "MQ_1_MS_U_Neig",
        "MQ_1_MS_U_Schieb",
        "MQ_2_MS_U_MI_L_o",
        "MQ_2_Ms_U_Li_Int_u",
        "MQ_2_MS_O_MI_L_u",
        "MQ_2_Ms_U_Re_Int_u",
        "MQ_2_MS_U_Neig",
        "MQ_3_MS_U_MI_L_o",
        "MQ_3_Ms_U_Li_Int_u",
        "MQ_3_MS_O_MI_L_u",
        "MQ_3_Ms_U_Re_Int_u",
        "MQ_3_MS_U_Neig",
        "MQ_4_MS_U_MI_L_o",
        "MQ_4_Ms_U_Li_Int_u",
        "MQ_4_MS_O_MI_L_u",
        "MQ_4_Ms_U_Re_Int_u",
        "MQ_4_MS_U_Neig",
        "MQ_5_MS_U_Neig",
        "MQ_5_MS_U_Schieb",
    ]

    def __init__(
        self,
        zip_file_path: Path,
        output_base_path: Path,
        tmp_path: Path,
        scenario: PBimArtificialScenario,
    ):
        self._zip_file_path = zip_file_path
        self._output_base_path = output_base_path
        self._tmp_path = tmp_path
        self._scenario = scenario
        self._parser = PBimArtificialDataParser()
        self._identifier = os.getpid()

    def _filter_and_sort_files(self, file_names: List[str], scenario: str) -> List[str]:
        def _filter(name: str) -> bool:
            match = self.NAME_REGEX.match(name)
            if not match:
                return False
            return match.group(2) == scenario

        def _sort_key_extractor(name: str) -> int:
            return int(self.NAME_REGEX.match(name).group(3))

        return sorted(
            [name for name in file_names if _filter(Path(name).name)],
            key=lambda name: _sort_key_extractor(Path(name).name),
        )

    def _process_file(
        self, zip_file: zipfile.ZipFile, file_name: str
    ) -> Dict[str, List[Measurement]]:
        file_path = Path(zip_file.extract(file_name, self._tmp_path))
        data = self._parser.parse(file_path, channels=self.CHANNELS)
        file_path.unlink()
        return data

    def _compute_start_time(self, file_path: str) -> datetime.datetime:
        file_name = Path(file_path).name
        file_number = self.NAME_REGEX.match(file_name).group(3)
        return self.BASE_TIME + datetime.timedelta(hours=int(file_number))

    @staticmethod
    def _serialize_measurement(measurement: Measurement) -> bytes:
        return struct.pack("<qf", measurement.time, measurement.measurement)

    def _save_data(self, data: Dict[str, List[Measurement]], fs: Dict[str, BinaryIO]):
        for channel, measurements in data.items():
            f = fs[channel]
            serialized = b"".join(
                [self._serialize_measurement(m) for m in measurements]
            )
            f.write(serialized)

    @staticmethod
    def _group_files(sorted_file_names: List[str]) -> List[List[str]]:
        return [
            sorted_file_names[i : i + 24] for i in range(0, len(sorted_file_names), 24)
        ]

    def _make_output_path(self, measurement_time: datetime.datetime) -> Path:
        return (
            self._output_base_path
            / f"{measurement_time.year}"
            / f"{measurement_time.month:02d}"
            / f"{measurement_time.day:02d}"
        )

    @staticmethod
    def _prepare_file_handles(
        output_path: Path, channels: List[str]
    ) -> Dict[str, BinaryIO]:
        output_path.mkdir(parents=True, exist_ok=True)
        return {
            channel: open(output_path / f"{channel}.dat", "wb") for channel in channels
        }

    def process(self):
        self._tmp_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self._zip_file_path) as zip_file:
            relevant_file_names = self._filter_and_sort_files(
                zip_file.namelist(), self._scenario
            )
            groups = self._group_files(relevant_file_names)
            LOGGER.info(
                f"Found {len(relevant_file_names)} relevant files corresponding to {len(groups)} days.",
                self._identifier,
            )
            for i, group in enumerate(groups):
                LOGGER.info(f"Processing group ({i+1}/{len(groups)})", self._identifier)
                measurement_time = self._compute_start_time(group[0])
                data = self._process_file(zip_file, group[0])
                output_path = self._make_output_path(measurement_time)
                file_handles = self._prepare_file_handles(
                    output_path, list(data.keys())
                )
                self._save_data(data, file_handles)
                for j, file_name in enumerate(group[1:]):
                    LOGGER.info(
                        f"Processing file '{file_name}' ({j+1}/{len(group)})",
                        self._identifier,
                    )
                    data = self._process_file(zip_file, file_name)
                    self._save_data(data, file_handles)
                for file_handle in file_handles.values():
                    file_handle.close()
