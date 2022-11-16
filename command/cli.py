from loguru import logger
from command.aliased_group import AliasedGroup
from command.evaluate import evaluate
from command.train import train

import click

__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == '__main__':
    cli()
