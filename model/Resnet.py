
from asyncio.windows_events import NULL
from turtle import forward
from typing import Tuple
import torch
import torch.nn as nn
from collections import OrderedDict

class _conv3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, out_channel, 3, stride=2)),
            ("BN", nn.BatchNorm2d(out_channel)),
            ("relu", nn.ReLU()),
            ("conv1", nn.Conv2d(out_channel, out_channel, 3, stride=2)),
            ("BN", nn.BatchNorm2d(out_channel))
        ]))
        self.conv1x1 = nn.Conv2d(in_channel, out_channel) if in_channel != out_channel else NULL

    def forward(self, x):
        y = nn.functional.relu(self.features(x) + x if self.conv1x1 == NULL else self.conv1x1(x))
        return y

class _conv1x3x1(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, mid_channel, 1)),
            ("BN", nn.BatchNorm2d(mid_channel)),
            ("relu", nn.ReLU()),
            ("conv1", nn.Conv2d(mid_channel, mid_channel, 3, stride=2)),
            ("BN", nn.BatchNorm2d(mid_channel)),
            ("relu", nn.ReLU()),
            ("conv1", nn.Conv2d(mid_channel, out_channel, 1)),
            ("BN", nn.BatchNorm2d(out_channel))
        ]))
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1) if in_channel != out_channel else NULL

    def forward(self, x):
        y = nn.functional.relu(self.features(x) + x if self.conv1x1 == NULL else self.conv1x1(x))
        return y


class Resnet(nn.Module):
    def __init__(self, init_channel = 64, block_config:Tuple = (), deep:bool = False, class_num = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("7x7conv", nn.Conv2d(3, init_channel, kernel_size=7, stride=2)),
            ("BN", nn.BatchNorm2d(init_channel)),
            ("relu", nn.ReLU()),
            ("maxpool", nn.MaxPool2d(3,stride=2))
        ]))
        channels = init_channel
        for i, num in enumerate(block_config):
            block = _conv3x3(init_channel, channels) if not deep else _conv1x3x1(init_channel, channels, channels*4)
            init_channel = channels if not deep else channels*4
            channels *= 2
            self.features.add_module("block%d" % i, block)

        self.features.add_module("avgpool", nn.AdaptiveAvgPool2d((1,1)))

        self.fcn1000 = nn.Linear(512*(3*deep+1), class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.fcn1000(torch.flatten(x, 1))
        return x

def resnet18(class_num):
    model = Resnet(block_config=(2,2,2,2), class_num=class_num)
    return model

def resnet34(class_num):
    model = Resnet(block_config=(3,4,6,3), class_num=class_num)
    return model

def resnet50(class_num):
    model = Resnet(block_config=(3,4,6,3), deep=True, class_num=class_num)
    return model

def resnet101(class_num):
    model = Resnet(block_config=(3,4,23,3), deep=True, class_num=class_num)
    return model

def resnet152(class_num):
    model = Resnet(block_config=(3,8,36,3), deep=True, class_num=class_num)
    return model