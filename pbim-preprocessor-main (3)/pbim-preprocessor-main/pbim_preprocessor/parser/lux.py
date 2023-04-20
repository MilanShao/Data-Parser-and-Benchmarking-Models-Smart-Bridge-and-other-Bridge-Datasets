import datetime
from pathlib import Path
from typing import Tuple, List, IO

import numpy as np
import pandas as pd
from nptdms import TdmsFile


class LuxAccelerationParser:
    @staticmethod
    def parse(file_path: Path | IO) -> Tuple[List[str], List[np.ndarray]]:
        channels = []
        data = []
        with TdmsFile.open(file_path) as tdms_file:
            time = tdms_file.groups()[0].channels()[0].time_track()
            channels.append("time")
            data.append(time)
            for channel in tdms_file.groups()[0].channels():
                channels.append(channel.name)
                data.append(channel[:])

        return channels, data


class LuxTemperatureParser:
    FIRST_TEMPERATURE_COLUMN_INDEX = 18

    @staticmethod
    def _make_time_stamp_in_ns(value: datetime.time) -> float:
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1e6
        ) * 1e9

    def parse(self, file: Path | IO) -> np.ndarray:
        df = pd.read_excel(file)
        df["average_temperature"] = df.values[
            :, self.FIRST_TEMPERATURE_COLUMN_INDEX :
        ].mean(axis=1)
        df["time"] = (
            df["Date"].values.astype(float)
            + df["Time (UTC)"].apply(self._make_time_stamp_in_ns).values.astype(float)
        ) / 1e6
        return df[["time", "average_temperature"]].values
