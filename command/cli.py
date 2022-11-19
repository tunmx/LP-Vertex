from loguru import logger
from command.aliased_group import AliasedGroup
from command.train import train
from command.evaluate import evaluate
from command.test import test
from command.make_dataset import make
from command.data_transform import transform
from command.visual_model import visual
import click

__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(test)
cli.add_command(make)
cli.add_command(transform)
cli.add_command(visual)

if __name__ == '__main__':
    cli()
