import torch
from data import LabelMeDataset
from model.mobile_vertex import MobileNetV2
from torch.utils.data import DataLoader
from trainer.task import TrainTask
from loguru import logger

train_dir = "oinbagCrawler_vertex_train/train"
val_dir = "oinbagCrawler_vertex_train/val"
batch_size = 192

train_dataset = LabelMeDataset(train_dir, mode='train', is_show=False)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = LabelMeDataset(val_dir, mode='val', is_show=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

logger.info(f"Training Dataset Total: {len(train_dataset)}")
logger.info(f"Verification Dataset Total: {len(val_dataset)}")

net = MobileNetV2()

lr_schedule_option = dict(name='ExponentialLR', gamma=0.1)
optimizer_option = dict(name='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

task_option = dict(model=net, save_dir='save_dir_t2', loss_func='mse_loss', lr_schedule_option=lr_schedule_option,
                   optimizer_option=optimizer_option, weight_path=None)

task = TrainTask(**task_option)

task.training(train_dataloader, val_dataloader, epoch_num=100, is_save=True)

if __name__ == '__main__':
    pass

