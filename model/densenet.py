from turtle import forward
from typing import OrderedDict
import torch

class Densenet():
    def __init__(self, out_channel):
        super().__init__()
        self.features = torch.nn.Sequential(OrderedDict([
            ("conv0", torch.nn.Conv2d(3, out_channel, 7, stride=2)),
            ("BN", torch.nn.BatchNorm2d(out_channel)),
            ()
        ]))

    def forward(self, x):
        
        pass