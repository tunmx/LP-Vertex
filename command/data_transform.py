import os
import click
import tqdm
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.data import get_dataset
from breezevertex.data import Pipeline
from torch.utils.data import DataLoader
import torch

__all__ = ['data_transform']


@click.command(help='Evaluation')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-data', '--data', default=None, type=click.Path())
@click.option("-show", "--show", is_flag=True, type=click.BOOL, )
def data_transform(config_path, data, show):
    logger.info("data_transform")
    cfg = load_cfg(config_path)
    # load data
    data_cfg = cfg.data
    if data:
        data_cfg.val.option.img_path = data
    # build val dataset
    transform = Pipeline(**data_cfg.pipeline)
    dataset = get_dataset(data_cfg.val.name, transform=transform, **data_cfg.val.option)
    batch_size = data_cfg.val.batch_size
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    transform_data = tqdm.tqdm(dataloader)
    data = next(iter(transform_data))
    print(data.shape)


if __name__ == '__main__':
    data_transform()
