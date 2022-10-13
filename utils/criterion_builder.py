import torch
def build_criterion(type):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion