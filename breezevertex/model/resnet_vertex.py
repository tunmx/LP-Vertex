import torch
import torch.nn as nn
import torch.nn.functional as f
from .backbone import ResNet


class ResNetVertex(nn.Module):

    def __init__(self, depth=50):
        super().__init__()
        self.backbone = ResNet(depth)
        self.pool1 = nn.AvgPool2d(4, stride=1)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.relu6 = nn.ReLU6()
        self.prelu1 = nn.PReLU()
        self.kps = nn.Linear(128, 8)

    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.prelu1(x)
        x = self.kps(x)

        return x


if __name__ == '__main__':
    net = ResNetVertex()
    print(net)

    x = torch.rand(1, 3, 112, 112)
    # print(x.shape)
    y = net(x)
    print(y.shape)
