from breezevertex.data import LabelMeDataset
from breezevertex.model.mobile_vertex import MobileVertex
from torch.utils.data import DataLoader
from breezevertex.trainer.task import TrainTask
from loguru import logger

train_dir = "oinbagCrawler_vertex_train/train"
val_dir = "oinbagCrawler_vertex_train/val"
batch_size = 128

train_dataset = LabelMeDataset(train_dir, mode='train', is_show=False)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = LabelMeDataset(val_dir, mode='val', is_show=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

logger.info(f"Training Dataset Total: {len(train_dataset)}")
logger.info(f"Verification Dataset Total: {len(val_dataset)}")

net = MobileVertex(width_mult=0.5)

# 暂时无用
lr_schedule_option = dict(name='ReduceLROnPlateau', mode='min', factor=0.5, patience=5, verbose=True, )
optimizer_option = dict(name='Adam', lr=0.001, )

task_option = dict(model=net, save_dir='save_dir_mnetv2_half', loss_func='mse_loss', lr_schedule_option=lr_schedule_option,
                   optimizer_option=optimizer_option, weight_path=None)

task = TrainTask(**task_option)

task.training(train_dataloader, val_dataloader, epoch_num=150, is_save=True)

if __name__ == '__main__':
    pass

