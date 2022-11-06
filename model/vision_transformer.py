import torch
import torch.nn as nn
from torchvision.models import vision_transformer


class Net(nn.Module):
    def __init__(self, num_classes:int, pretrain):
        super().__init__()
        self.fc = nn.Linear(1000, num_classes)
        self.vit = vision_transformer.VisionTransformer(image_size=224,
                                                        patch_size=16,
                                                        num_layers=12,
                                                        num_heads=12,
                                                        hidden_dim=768,
                                                        mlp_dim=3072)
        if pretrain:
            self.vit.load_state_dict(torch.load(pretrain))

    def forward(self, x):
        _1000 = self.vit(x)
        y = self.fc(_1000)
        return y

    def trainable_parameters(self):
        return [list(self.vit.parameters()), list(self.fc.parameters())]
