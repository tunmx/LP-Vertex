import numpy as np
from loguru import logger
import os
import tqdm
from .base import VertexDatasetBase


class TxtLineDataset(VertexDatasetBase):
    def _load_data(self, img_path: str, labels_path: str) -> list:
        pass

    def _load_label(self, path: str) -> dict:
        pass
