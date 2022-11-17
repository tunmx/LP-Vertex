import yaml
from easydict import EasyDict as edict


def load_cfg(config_path: str) -> edict:
    with open(config_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(data_dict)

    return cfg
