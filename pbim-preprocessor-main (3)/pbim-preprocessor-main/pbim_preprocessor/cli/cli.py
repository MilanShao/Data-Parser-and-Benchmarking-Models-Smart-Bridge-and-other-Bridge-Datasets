import os

import click

from pbim_preprocessor.cli.assemble import assemble
from pbim_preprocessor.cli.generate import generate
from pbim_preprocessor.cli.merge import merge
from pbim_preprocessor.cli.parse import parse
from pbim_preprocessor.cli.plot import plot
from pbim_preprocessor.cli.postprocess import postprocess
from pbim_preprocessor.cli.process import process
from pbim_preprocessor.cli.process_artificial import process_artificial
from pbim_preprocessor.utils import LOGGER


@click.group()
def cli():
    LOGGER.set_debug(
        os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "y", "yes")
    )
    LOGGER.debug("Debug mode enabled")


cli.add_command(parse)
cli.add_command(plot)
cli.add_command(process)
cli.add_command(assemble)
cli.add_command(generate)
cli.add_command(merge)
cli.add_command(process_artificial)
cli.add_command(postprocess)
