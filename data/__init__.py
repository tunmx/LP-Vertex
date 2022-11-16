from .dataset import LabelMeDataset
from .dataset.base import VertexDatasetBase

_dataset_map_ = dict(LabelMeDataset=LabelMeDataset, )


def get_dataset(name: str, **option) -> VertexDatasetBase:
    assert name in _dataset_map_.keys()
    return _dataset_map_[name](**option)
