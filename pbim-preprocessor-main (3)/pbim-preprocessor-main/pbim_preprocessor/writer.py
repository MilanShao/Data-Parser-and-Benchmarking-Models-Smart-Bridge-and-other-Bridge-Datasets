import abc
import struct
from pathlib import Path
from typing import List, Dict


class Writer(abc.ABC):
    def __init__(self, path: Path, headers: List[str]):
        self._path = path
        self._headers = headers

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def write_step(self, data: Dict[str, float], time: int) -> None:
        pass


class CsvWriter(Writer):
    def __init__(self, path: Path, headers: List[str], delimiter=","):
        super().__init__(path, headers)
        self._delimiter = delimiter
        self._f = None

    def __enter__(self):
        self._f = open(self._path, "w")
        self._f.write(self._delimiter.join(["Time"] + self._headers) + "\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    def write_step(self, data: Dict[str, float], time: int) -> None:
        line = self._delimiter.join([f"{data[header]:.6f}" for header in self._headers])
        self._f.write(line + "\n")


class BinaryWriter(Writer):
    def __init__(self, path: Path, headers: List[str], time_byte_size: int = 4):
        super().__init__(path, headers)
        self._f = None
        self._time_byte_size = time_byte_size

    def __enter__(self):
        self._f = open(self._path, "wb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    def write_step(self, data: Dict[str, float], time: int) -> None:
        values = [data[channel] for channel in self._headers]
        time_format = "q" if self._time_byte_size == 8 else "i"
        binary_format = f"<{time_format}{len(values)}f"
        self._f.write(struct.pack(binary_format, time, *values))
