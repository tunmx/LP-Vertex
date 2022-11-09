import torch
import torch.nn as nn
import torch.nn.functional as f
from .backbone import ShuffleNetV2


class ShuffleVertex(nn.Module):

    def __init__(self, model_size='1.5x', pretrain=True, with_last_conv=False):
        super().__init__()
        self.backbone = ShuffleNetV2(model_size='1.5x', pretrain=True, with_last_conv=True)
        self.conv1 = nn.Conv2d(1024, 128, 1, 1, 0, bias=False)
        self.pool1 = nn.AvgPool2d(4, stride=1)
        self.linear = nn.Linear(128, 8)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu6(x)
        return x


if __name__ == '__main__':
    net = ShuffleVertex()
    print(net)

    x = torch.rand(1, 3, 112, 112)
    print(x.shape)
    y = net(x)
    print(y.shape)
