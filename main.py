from pickletools import optimize
import torch
import argparse
import yaml
from utils.dataset_builder import build_dataset
from utils.model_builder import build_model
from utils.criterion_builder import build_criterion
from asyncio.windows_events import NULL
from tqdm import tqdm
import model.dense_clasifier as dense_clasifier
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def load_config(args):
    with open(args.train_conf, 'r') as f:
        config = yaml.safe_load(f)
        model_conf = config['model']
        data_conf = config['dataset']
        train_conf = config['train']
    return model_conf, data_conf, train_conf


def train(train_config, model, dataloader, testloader, device):
    learning_rate = float(train_config['leaning_rate'])
    criterion = build_criterion(train_config['criterion'])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    for epoch in range(int(train_config['epoch'])):
        total_loss = 0.0
        model.train()
        for i_batch, img in enumerate(tqdm(dataloader)):
            x, y = img[0].to(device), img[1].to(device)
            optimizer.zero_grad()
    
            pre = model(x)
            loss = criterion(pre, y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        best_acc = 0.0
        print('epoch-%d    '%(epoch) + 'total loss:' + str(total_loss))
        if epoch % 5 == 0:
            model.eval()
            acc = 0
            val_num = 0
            best_acc = 0
            with torch.no_gard():
                for val_data in testloader:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                    val_num += val_labels.size(0)
                val_accurate = acc / val_num
                writer.add_scalar("Acc/test", val_accurate, epoch)
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(model.state_dict(), './weights/dense_ep' + epoch + '.pt')
                print('epoch-%d    '%(epoch) + 'val acc:' + str(val_accurate))
                

def once_test(model_pth, device, valset):
    model = dense_clasifier.densenet161(15)
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    model.to(device)
    acc = 0
    val_num = 0
    best_acc = 0
    with torch.no_gard():
        for val_data in valset:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
            val_num += val_labels.size(0)
        val_accurate = acc / val_num
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        print(val_accurate)


def main(args):
    model_config, data_config, train_config= load_config(args)
    imgloader, valset = build_dataset(data_config)
    model = build_model(model_config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(train_config, model, imgloader, valset, device)

    once_test('./weights/dense_ep95', device, valset)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_conf', type = str, default='./config/train_conf.yaml')
    args = parser.parse_args()
    main(args)