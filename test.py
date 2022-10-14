import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms,utils
from torch.utils.data import DataLoader
import model.dense as dense

my_trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


valset = ImageFolder('./data/test', transform=my_trans)
val = DataLoader(valset, batch_size = 64 , shuffle=False, num_workers=0)


model_Path = './weights'
model = dense.densenet161(15)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

for model_pth in os.listdir(model_Path):
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    acc = 0
    val_num = 0
    best_acc = 0
    with torch.no_grad():  
        for val_data in val:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
            val_num += val_labels.size(0)
        val_accurate = acc / val_num
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        print(model_pth + '---' + val_accurate)

