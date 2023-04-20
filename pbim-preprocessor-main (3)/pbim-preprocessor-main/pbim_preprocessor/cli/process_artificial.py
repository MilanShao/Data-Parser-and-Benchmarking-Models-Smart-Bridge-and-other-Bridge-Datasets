from pathlib import Path

import click

from pbim_preprocessor.processor.pbim import (
    PBimArtificialDataProcessor,
    PBimArtificialScenario,
)


@click.command()
@click.argument("zip_file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_base_path", type=click.Path(path_type=Path, writable=True))
@click.argument("scenario", type=click.Choice(["N", "S0", "S1", "S2", "S3"]))
@click.option("--tmp-path", default=Path("/tmp/out"), type=click.Path(path_type=Path))
def process_artificial(
    zip_file_path: Path,
    output_base_path: Path,
    scenario: PBimArtificialScenario,
    tmp_path: Path,
):
    processor = PBimArtificialDataProcessor(
        zip_file_path=zip_file_path,
        output_base_path=output_base_path,
        tmp_path=tmp_path,
        scenario=scenario,
    )
    processor.process()
