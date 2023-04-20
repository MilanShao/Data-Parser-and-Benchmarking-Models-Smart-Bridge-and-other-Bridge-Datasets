from pathlib import Path
from typing import List, Generator, Dict, Any, Optional, TextIO, Tuple


class GrandStandAssembler:
    def __init__(self, path: Path):
        self._path = path

    def assemble(
        self, scenario: str, channels: List[str]
    ) -> Generator[Dict[str, float], Any, None]:
        path = self._make_path(scenario)
        with open(path, "r") as f:
            line, channel_order = self._parse_channel_order(f)
            if not channel_order:
                raise ValueError(f"Failed to parse metadata for scenario {scenario}.")
            yield self._annotate(self._parse_line(line), channel_order, 0)
            for i, line in enumerate(f):
                yield self._annotate(self._parse_line(line), channel_order, i + 1)

    @staticmethod
    def _annotate(
        data: List[float],
        channel_order: List[str],
        measurement_index: int,
        desired_channels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        time_index = channel_order.index("time")
        data[time_index] = measurement_index
        return {
            channel: value
            for channel, value in zip(channel_order, data)
            if desired_channels is None or channel in desired_channels
        }

    @staticmethod
    def _parse_line(line: str) -> List[float]:
        return [float(x) for x in line.split("\t")]

    @staticmethod
    def _parse_channel_order(f: TextIO) -> Tuple[str, Optional[List[str]]]:
        order = None
        while (line := f.readline()).startswith('"'):
            if line.startswith('"Y Axis DOFS"'):
                order = ["time"] + [
                    "Joint " + x.strip().replace('"', "") for x in line.split("\t")[1:]
                ]
        return line, order

    def _make_path(self, scenario: str) -> Path:
        return self._path / f"{scenario}.txt"
