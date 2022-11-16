import click
import os
import sys
import platform
from loguru import logger

__all__ = ['train']


@click.command(help='Training')
@click.argument('config_path')
def train(config_path, ):
    logger.info("training")
    pass
