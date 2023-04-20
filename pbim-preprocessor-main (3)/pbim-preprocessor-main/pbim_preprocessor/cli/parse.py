import json
from pathlib import Path

import click

from pbim_preprocessor.cli import CHANNELS_TO_PROCESS
from pbim_preprocessor.parser.pbim import PBimRealDataParser


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, file_okay=False, path_type=Path)
)
@click.argument("name")
def parse(path: Path, output_path: Path, name: str):
    """
    Parse data from a Job specified by NAME located in the directory specified by PATH. Results are stored in the directory
    specified by OUTPUT_PATH.
    """
    output_path.mkdir(exist_ok=True, parents=True)
    parser = PBimRealDataParser()
    data = parser.parse(path, name, CHANNELS_TO_PROCESS)
    for name, parsed_channel in data.items():
        with open(output_path / f"{name}.json", "w") as f:
            json.dump(
                {
                    "metadata": parsed_channel.channel_header.to_dict(),
                    "data": [
                        measurement.measurement
                        for measurement in parsed_channel.measurements
                    ],
                    "time": [
                        measurement.time for measurement in parsed_channel.measurements
                    ],
                },
                f,
            )
