import click
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.model import build_model

__all__ = ['visual']


@click.command(help='Visual model.')
@click.argument('config_path', type=click.Path(exists=True))
def visual(config_path):
    logger.info("evaluation")
    cfg = load_cfg(config_path)
    print(cfg)
    # build training model
    model_cfg = cfg.model
    net = build_model(model_cfg.name, **model_cfg.option)
    print(net)
