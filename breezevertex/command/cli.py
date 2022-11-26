from breezevertex.command.aliased_group import AliasedGroup
from breezevertex.command.train import train
from breezevertex.command.evaluate import evaluate
from breezevertex.command.test import test
from breezevertex.command.make_dataset import make
from breezevertex.command.data_transform import transform
from breezevertex.command.visual_model import visual
from breezevertex.command.generate import generate
from breezevertex.command.export import export
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
cli.add_command(generate)
cli.add_command(export)

if __name__ == '__main__':
    cli()
