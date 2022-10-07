from turtle import forward
import torch

class Densenet():
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, out_channel, 7, kernel_size=7, stride=2)
        self.maxpool = torch.nn.MaxPool2d()
    def forward(self, x):
        
        pass