import torch
import torch.nn as nn
import torch.nn.functional as f
from backbone import MobileNetV2


class MobileVertex(nn.Module):

    def __init__(self, width_mult=1., last_channel=1280):
        super().__init__()
        self.backbone = MobileNetV2(width_mult=width_mult, last_channel=last_channel)
        self.pool1 = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(1280, 128)
        # self.relu6 = nn.ReLU6()
        self.prelu1 = nn.PReLU()
        self.kps = nn.Linear(128, 8)

    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.prelu1(x)
        x = self.kps(x)

        return x


if __name__ == '__main__':
    net = MobileVertex()
    print(net)

    x = torch.rand(1, 3, 112, 112)
    print(x.shape)
    y = net(x)
    print(y.shape)
