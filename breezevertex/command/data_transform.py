import os
import click
import cv2
import tqdm
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.utils.image_tools import visual_images
from breezevertex.data import get_dataset
from breezevertex.data import Pipeline
from torch.utils.data import DataLoader

__all__ = ['transform']


@click.command(help='View the transformed data set.')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-data', '--data', default=None, type=click.Path())
def transform(config_path, data):
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
    images_tensor, kps_tensor = next(iter(transform_data))
    _, _, h, w = images_tensor.shape
    draw_images = visual_images(images_tensor, kps_tensor, w, h, swap=False)
    for img in draw_images:
        cv2.imshow("img", img)
        cv2.waitKey(0)



if __name__ == '__main__':
    transform()
