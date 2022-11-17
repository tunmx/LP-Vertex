from loguru import logger
import os
import tqdm
import json
from .base import VertexDatasetBase
import numpy as np


class LabelMeDataset(VertexDatasetBase):

    def _load_label(self, path: str) -> dict:
        with open(path, "r") as f:
            result = json.load(f)

        return result

    def _load_data(self, img_path, labels_path) -> list:
        results = list()
        path_list = list()
        assert type(img_path) == type(labels_path)
        if isinstance(img_path, str):
            path_list.append([img_path, labels_path])
        elif isinstance(img_path, list):
            assert len(img_path) == len(labels_path)
            for idx, p in enumerate(img_path):
                path_list.append([p, labels_path[idx]])
        else:
            pass
        for x_path, y_path in path_list:
            list_ = [os.path.join(x_path, item) for item in os.listdir(x_path) if
                     item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
            logger.info("Data Loading...")
            for idx, path in enumerate(tqdm.tqdm(list_)):
                basename = os.path.basename(path)
                label = "".join(basename.split('.')[:-1]) + ".json"
                label_full = os.path.join(y_path, label)
                if os.path.exists(label_full):
                    label_data = self._load_label(label_full)
                    if label_data.get("shapes")[0]:
                        polyline = np.asarray(label_data['shapes'][0]['points'])
                        dic = dict(image=path, label=polyline)
                        results.append(dic)

        return results
