import torch
def build_criterion(type):
    if type == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    return criterion