import os

import click
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.model import build_model
from breezevertex.data import get_dataset
from breezevertex.data import Pipeline
from breezevertex.trainer import TrainTask
from torch.utils.data import DataLoader

__all__ = ['train']


@click.command(help='Perform model training tasks.')
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path, ):
    cfg = load_cfg(config_path)
    print(cfg)
    # create folder
    os.makedirs(cfg.save_dir, exist_ok=True)
    # build training model
    model_cfg = cfg.model
    net = build_model(model_cfg.name, **model_cfg.option)
    # build training dataset
    data_cfg = cfg.data
    transform = Pipeline(**data_cfg.pipeline)
    train_dataset = get_dataset(data_cfg.train.name, transform=transform, **data_cfg.train.option)
    train_batch_size = data_cfg.train.batch_size
    # build val dataset
    val_dataset = get_dataset(data_cfg.val.name, transform=transform, **data_cfg.val.option)
    val_batch_size = data_cfg.val.batch_size
    logger.info(f"Training Dataset Total: {len(train_dataset)}")
    logger.info(f"Verification Dataset Total: {len(val_dataset)}")
    # create dataloader
    trainer_cfg = cfg.trainer
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=trainer_cfg.worker_num)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=trainer_cfg.worker_num)
    # build trainer
    task_option = dict(model=net, save_dir=cfg.save_dir, loss_func=model_cfg.loss.name,
                       lr_schedule_option=trainer_cfg.schedule, optimizer_option=trainer_cfg.optimizer, wandb_cfg=cfg.wandb,
                       weight_path=model_cfg.weight)
    task = TrainTask(**task_option)
    # run training
    task.training(train_dataloader, val_dataloader, epoch_num=trainer_cfg.epoch_num, is_save=True)


if __name__ == '__main__':
    train()
