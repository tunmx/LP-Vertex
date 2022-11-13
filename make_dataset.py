import os
import random
import shutil

import cv2
import numpy as np
import tqdm

from data import LabelMeDataset


if __name__ == '__main__':
    dataset = LabelMeDataset("/Users/tunm/datasets/oinbagCrawler_vertex/data", mode='test', is_show=True)
    save_path = "/Users/tunm/datasets/oinbagCrawler_vertex_train"
    train_dir = os.path.join(save_path, "train")
    val_dir = os.path.join(save_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    list_ = dataset.data_list
    random.shuffle(list_)

    split_rate = 0.92
    train_num = int(len(dataset) * split_rate)
    train_set = list_[:train_num]
    val_set = list_[train_num:]

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

