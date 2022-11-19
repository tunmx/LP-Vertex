import os
import random
import shutil
from loguru import logger
import tqdm
import click

from breezevertex.data import LabelMeDataset

__all__ = ['make']


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.argument('save_path', type=click.Path())
@click.option('-split_rate', '--split_rate', default=0.95, type=float)
def make(path, save_path, split_rate):
    dataset = LabelMeDataset(path, mode='test', is_show=True)
    train_dir = os.path.join(save_path, "train")
    val_dir = os.path.join(save_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    list_ = dataset.data_list
    random.shuffle(list_)

    train_num = int(len(dataset) * split_rate)
    train_set = list_[:train_num]
    val_set = list_[train_num:]
    logger.info(f"train: {train_set}/{train_num}")
    logger.info(f"val: {val_set}/{train_num}")

    for item in tqdm.tqdm(train_set):
        img_path = item['image']
        label_path = "".join(img_path.split(".")[:-1]) + ".json"
        dst_img = os.path.join(train_dir, os.path.basename(img_path))
        dst_label = os.path.join(train_dir, os.path.basename(label_path))
        shutil.copy(img_path, dst_img)
        shutil.copy(label_path, dst_label)

    for item in tqdm.tqdm(val_set):
        img_path = item['image']
        label_path = "".join(img_path.split(".")[:-1]) + ".json"
        dst_img = os.path.join(val_dir, os.path.basename(img_path))
        dst_label = os.path.join(val_dir, os.path.basename(label_path))
        shutil.copy(img_path, dst_img)
        shutil.copy(label_path, dst_label)


if __name__ == '__main__':
    make()
