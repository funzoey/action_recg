from turtle import forward
from typing import OrderedDict
import torch

class _denselayer():
    def __init__(self):
        pass

class Densenet():
    def __init__(self, out_channel):
        super().__init__()
        self.features = torch.nn.Sequential(OrderedDict([
            ("conv0", torch.nn.Conv2d(3, out_channel, kernel_size = 7, stride=2)),
            ("BN", torch.nn.BatchNorm2d(out_channel)),
            ("relu", torch.nn.ReLU()),
            ("maxpool", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))


    def forward(self, x):
        
        pass