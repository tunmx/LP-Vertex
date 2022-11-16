from .shuffle_vertex import ShuffleVertex
from .mobile_vertex import MobileVertex
from .resnet_vertex import ResNetVertex
import torch.nn as nn

_models_list_ = dict(ShuffleVertex=ShuffleVertex, MobileVertex=MobileVertex, ResNetVertex=ResNetVertex)


def build_model(name: str, **option) -> nn.Module:
    assert name in _models_list_.keys()
    return _models_list_[name](**option)
