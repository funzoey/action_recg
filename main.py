from pickletools import optimize
import torch
import argparse
import yaml
from utils.dataset_builder import build_dataset
from utils.model_builder import build_model
from asyncio.windows_events import NULL
from tqdm import tqdm
import model.dense as dense

def load_config(args):
    with open(args.train_conf, 'r') as f:
        config = yaml.safe_load(f)
        model_conf = config['model']
        data_conf = config['dataset']
        train_conf = config['train']
    return model_conf, data_conf, train_conf


def train(train_config, model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(int(train_config['epoch'])):
        for i_batch, img in enumerate(tqdm(dataloader)):
            x, y = img[0].to(device), img[1].to(device)
            optimizer.zero_grad()
            
            pre = model(x)
            loss = criterion(pre, y)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), './weights/dense_ep' + epoch + '.pt')



def test(model_pth, device):
    model = dense.densenet161(15)
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    acc = 0.0

    # with torch.no_gard():  # 没有梯度
    #     for val_data in validate_loader:
    #         val_images, val_labels = val_data
    #         outputs = model(val_images.to(device))
    #         predict_y = torch.max(outputs, dim=1)[1]
    #         acc += (predict_y == val_labels.to(device)).sum().item()
    #     val_accurate = acc / val_num
    #     if val_accurate > best_acc
    #         best_acc = val_accurate
    #         torch.save(net.state_dict(), save_path)
    #     print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / step, val_accurate))


def main(args):
    model_config, data_config, train_config= load_config(args)
    imgloader, val = build_dataset(data_config)
    model = build_model(model_config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(train_config, model, imgloader, device)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_conf', type = str, default='./config/train_conf.yaml')
    args = parser.parse_args()
    main(args)