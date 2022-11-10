import copy
import os
from loguru import logger
import torch
from torchmetrics import Accuracy
from tqdm import tqdm
from model.loss import loss_function
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.utils.data import DataLoader
from loguru import logger


class TrainTask(object):
    def __init__(self, model, save_dir, loss_func, optimizer_option, lr_schedule_option, weight_path=None):
        self.save_dir = save_dir
        self.model = model
        self.task_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # 设施学习率下降策略
        self.loss_func = loss_function(loss_func)
        self.optimizer_option = optimizer_option
        self.lr_schedule_option = lr_schedule_option
        self.optimizer, self.lr_scheduler = None, None
        self._configure_optimizers()
        self.best_accuracy = 0.0
        self.best_loss = 1000.0
        self.time_tag = datetime.datetime.now()

        if weight_path:
            self._load_pretraining_model(weight_path)
        self.model = self.model.to(self.task_device)

        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(self.save_dir)

    def _load_pretraining_model(self, weight_path=None):
        # 加载预训练模型
        logger.info('loading pretraining model {}'.format(weight_path))
        pretrained_dict = torch.load(weight_path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # TODO 加载预训练权重，冻结所有层，只训练最后的分类头，如果新增此方式，
        # #冻结最后一层
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
        # missing_keys, unexpected_keys = self.model.load_state_dict(
        #     pretrained_dict, strict=False)
        # for param in self.model.features.parameters():
        #     param.requires_grad = False

    def training(self, train_data: DataLoader, val_data: DataLoader, epoch_num: int, is_save=True):
        logger.info("Start training.")
        logger.info(f"Training Epochs Total: {epoch_num}")
        logger.info(f"Training Result Save to {self.save_dir}")
        for epoch in range(epoch_num):
            # 训练集
            train_loss = self.train_one_epoch(train_data, epoch, epoch_num)
            # logger.info(f"Train Epoch[{epoch + 1}/{epoch_num}] train_loss: {train_loss}")
            # 验证集
            val_loss = self.validation_one_epoch(val_data)
            logger.info(f"Train Epoch[{epoch + 1}/{epoch_num}] val_loss: {val_loss}")
            if is_save:
                self.save_model(val_loss, epoch, mode='min')

            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)
            self.writer.add_scalar('lr/epoch', self.lr_scheduler.get_last_lr()[0], epoch)

        logger.info("This training is completed, a total of {} rounds of training, training time: {} minutes" \
                    .format(epoch_num, (datetime.datetime.now() - self.time_tag).seconds // 60))

    def train_one_epoch(self, train_data, epoch, epochs_total):
        global loss_
        self.model.training()
        # train_acc = 0.0
        train_bar = tqdm(train_data)
        for step, data in enumerate(train_bar):
            samples, labels = data
            samples = samples.to(self.task_device)
            labels = labels.to(self.task_device)
            self.optimizer.zero_grad()
            outputs = self.model(samples.to(self.task_device))
            loss_ = self.loss_func(outputs, labels.to(self.task_device))
            loss_.backward()

            self.optimizer.step()
            self.scheduler.step()

            train_bar.set_description('Epoch: [{}/{}] loss: {:.3f}: lr: {:.3f}'.format(epoch + 1, epochs_total, loss_,
                                                                                       self.lr_scheduler.get_last_lr()))

        return loss_.item()

    def validation_one_epoch(self, val_data):
        self.model.eval()
        val_loss = 0.0
        # val_acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_data)
            for step, data in enumerate(val_bar):
                val_images, val_labels = data
                # val_images[0] = np.
                outputs = self.model(val_images.to(self.task_device))
                loss = self.loss_func(outputs, val_labels.to(self.task_device))
                val_loss += loss.item()

                val_bar.set_description(
                    'Val: loss: {:.3f}'.format(val_loss / (step + 1)))

        return val_loss / len(val_bar)

    def _configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.optimizer_option)
        logger.info("loading optimizer {}".format(optimizer_cfg.get('name')))
        name = optimizer_cfg.pop('name')
        build_optimizer = getattr(torch.optim, name)
        self.optimizer = build_optimizer(params=self.model.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.lr_schedule_option)
        name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(
            optimizer=self.optimizer, **schedule_cfg)

    def save_model(self, loss, epoch, mode='min'):
        torch.save(self.model.state_dict(), os.path.join(
            self.save_dir, 'last_t0.pth'))
        if mode == 'min':
            # 根据损失率最小保存
            if loss < self.best_loss:
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_dir, 'model_%d_loss%0.3f.pth' % (epoch + 1, loss)))
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_dir, 'best_model.pth'))
                self.best_loss = loss
