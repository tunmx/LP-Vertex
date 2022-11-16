import click
import os
import sys
import platform
from loguru import logger

__all__ = ['evaluate']


@click.command(help='evaluation running')
@click.argument('config_path')
@click.option("--weight", "-w", help="pre-train model path.")
def train(config_path, weight):
    logger.info("evaluation")
    pass
