import torch
import torch.nn as nn
import torch.nn.functional as f
from .backbone import MobileNetV2


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class MobileVertex(nn.Module):

    def __init__(self, width_mult=1., last_channel=1280, pool_pad=4):
        super().__init__()
        print('width_mult: ', width_mult)
        self.backbone = MobileNetV2(width_mult=width_mult, last_channel=last_channel)
        # self.conv_1x1 = conv_1x1_bn(1280, 512)
        self.pool1 = nn.AvgPool2d(pool_pad, stride=1)
        self.fc = nn.Linear(1280, 128)
        # self.relu6 = nn.ReLU6()
        self.prelu1 = nn.PReLU()
        self.kps = nn.Linear(128, 8)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.conv_1x1(x)
        # print('bb: ', x.shape)
        x = self.pool1(x)
        # print('pool: ', x.shape)
        x = x.view(x.size(0), -1)
        # print('view: ', x.shape)
        x = self.fc(x)
        # print('fc: ', x.shape)
        x = self.prelu1(x)
        x = self.kps(x)

        return x


if __name__ == '__main__':
    net = MobileVertex(0.5)
    # print(net)

    x = torch.rand(1, 3, 112, 112)
    # print(x.shape)
    y = net(x)
    print(y)
    print(y.shape)
