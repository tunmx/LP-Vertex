from abc import ABCMeta, abstractmethod, ABC
import cv2
import numpy as np
from torch.utils.data import Dataset
from data.transform import Pipeline


class VertexDatasetBase(Dataset, metaclass=ABCMeta):

    def __init__(self, img_path, labels_path=None, mode='train', transform=None, is_show=False):
        self.img_path = img_path
        self.is_show = is_show
        if labels_path:
            self.labels_path = labels_path
        else:
            self.labels_path = img_path
        self.mode = mode
        if transform:
            self.transform = transform
        else:
            self.transform = Pipeline()
        self.data_list = self._load_data(self.img_path, self.labels_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            return self._get_val_data(idx)
        else:
            return self._get_train_data(idx)

    def _load_data(self, img_path: str, labels_path: str) -> list:
        pass

    def _load_label(self, path: str) -> dict:
        pass

    def _data_to_tensor(self, image: np.ndarray, points: np.ndarray):
        height, width, _ = image.shape
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        # print(image.shape)
        points[:, 0] /= width
        points[:, 1] /= height
        points = points.reshape(-1)
        return image.astype(np.float32), points.astype(np.float32)

    def _get_train_data(self, idx: int) -> tuple:
        data = self.data_list[idx]
        image_path = data['image']
        label_kps = data['label']
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        x, y = self.transform(image, label_kps, mode='train')
        if not self.is_show:
            x, y = self._data_to_tensor(x, y)

        return x, y

    def _get_val_data(self, idx: int) -> tuple:
        data = self.data_list[idx]
        image_path = data['image']
        label_kps = data['label']
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        x, y = self.transform(image, label_kps, mode='val')
        if not self.is_show:
            x, y = self._data_to_tensor(x, y)

        return x, y
