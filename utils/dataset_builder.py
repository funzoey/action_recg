from torchvision.datasets import ImageFolder
from torchvision import transforms,utils
from torch.utils.data import DataLoader

my_trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_dataset(args:dict):
    data_path = args['path']
    imgset = ImageFolder(data_path, transform=my_trans)
    data = DataLoader(imgset, batch_size = 2, shuffle=True)
    return data