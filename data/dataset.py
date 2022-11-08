import json
from abc import ABCMeta, abstractmethod, ABC
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from .transform import Pipeline



def _load_label(path: str) -> dict:
    with open(path, "r") as f:
        result = json.load(f)

    return result


def _load_data(img_path, labels_path):
    results = list()
    list_ = [os.path.join(img_path, item) for item in os.listdir(img_path) if
             item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    for idx, path in enumerate(list_):
        basename = os.path.basename(path)
        label = "".join(basename.split('.')[:-1]) + ".json"
        label_full = os.path.join(labels_path, label)
        if os.path.exists(label_full):
            label_data = _load_label(label_full)
            if label_data.get("shapes")[0]:
                polyline = np.asarray(label_data['shapes'][0]['points'])
                dic = dict(image=path, label=polyline)
                results.append(dic)

    return results


class VertexDataset(Dataset, ABC):

    def __init__(self, img_path, labels_path=None, mode='train', transform=None):
        self.img_path = img_path
        if labels_path:
            self.labels_path = labels_path
        else:
            self.labels_path = img_path
        self.mode = mode
        if self.transform:
            self.transform = transform
        else:
            self.transform = Pipeline()
        self.data_list = _load_data(self.img_path, self.labels_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pass

    def _get_train_data(self, idx):
        data = self.data_list[idx]
        image_path = data['image']
        label_kps = data['label']
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        return self.transform(image, label_kps)

