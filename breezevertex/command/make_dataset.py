import os
import random
import shutil
from loguru import logger
import tqdm
import click
import json

from breezevertex.data import LabelMeDataset

__all__ = ['make']

CODE_OK = 0  # 通过
CODE_NOT_FOUND = 1  # 缺少标注文件
CODE_NOT_LABELED = 2  # 没有标注数据
CODE_NOT_MATCH = 3  # 标注数据不匹配


def check(img_path, label_path):
    if not os.path.exists(label_path):
        # logger.info(f"{label_path} 标注文件不存在!")
        return CODE_NOT_FOUND
    with open(label_path, 'r') as f:
        data = json.load(f)
    if not data['shapes']:
        # logger.info(f"{label_path} 缺少标注数据!")
        return CODE_NOT_LABELED
    # print(len(data['shapes'][0]['points']))
    if len(data['shapes'][0]['points']) != 4:
        # logger.info(f"{label_path} 数据标注不匹配!")
        return CODE_NOT_MATCH
    return CODE_OK


@click.command(help="Create a data set for training.")
@click.option('--path', type=click.Path(exists=True))
@click.option('--save_path', type=click.Path())
@click.option('-split_rate', '--split_rate', default=0.95, type=float)
@click.option('-suffix', '--suffix', type=str, default='json')
def make(path, save_path, split_rate, suffix):
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
        label_path = "".join(img_path.split(".")[:-1]) + f".{suffix}"
        dst_img = os.path.join(train_dir, os.path.basename(img_path))
        dst_label = os.path.join(train_dir, os.path.basename(label_path))
        if check(dst_img, dst_label) == CODE_OK:
            continue
        shutil.copy(img_path, dst_img)
        shutil.copy(label_path, dst_label)

    for item in tqdm.tqdm(val_set):
        img_path = item['image']
        label_path = "".join(img_path.split(".")[:-1]) + f".{suffix}"
        dst_img = os.path.join(val_dir, os.path.basename(img_path))
        dst_label = os.path.join(val_dir, os.path.basename(label_path))
        if check(dst_img, dst_label) == CODE_OK:
            continue
        shutil.copy(img_path, dst_img)
        shutil.copy(label_path, dst_label)


if __name__ == '__main__':
    make()
