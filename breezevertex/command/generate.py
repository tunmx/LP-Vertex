import os
import click
from loguru import logger
from breezevertex.utils.base import config_default

__all__ = ['generate']


@click.command(help='Create a new default config file.')
@click.option('-path', '--path', type=click.Path())
def generate(path):
    default_yml = os.path.join(path, 'vertex_resnet50_default.yml')
    with open(default_yml, 'w') as f:
        f.write(config_default)
    logger.info(f'Create a new default config file to {default_yml}')
    logger.info('You can modify the contents of the configuration file for training models and other operations.')


if __name__ == '__main__':
    generate()
