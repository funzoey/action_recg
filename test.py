import os
from torchvision.datasets import ImageFolder
from torchvision import transforms,utils
from torch.utils.data import DataLoader

my_trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_dataset():
    valset = ImageFolder(args['val_path'], transform=my_trans)
    val = DataLoader(valset, batch_size = 64 , shuffle=False, num_workers=0)
    return data, val

filePath = './weights'
os.listdir(filePath)

