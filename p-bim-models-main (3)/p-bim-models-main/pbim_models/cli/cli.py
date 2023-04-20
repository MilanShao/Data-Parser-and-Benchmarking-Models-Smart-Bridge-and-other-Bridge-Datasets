import click

from pbim_models.cli.evaluate import evaluate
from pbim_models.cli.predict import predict
from pbim_models.cli.predict_gan import predict_gan
from pbim_models.cli.sweep import sweep
from pbim_models.cli.train import train


@click.Group
def cli():
    pass


cli.add_command(train)
cli.add_command(sweep)
cli.add_command(evaluate)
cli.add_command(predict)
cli.add_command(predict_gan)
