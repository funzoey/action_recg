from pickletools import optimize
import torch
import argparse
import yaml
from utils.dataset_builder import build_dataset
from utils.model_builder import build_model
from asyncio.windows_events import NULL
from tqdm import tqdm

def load_config(args):
    with open(args.train_conf, 'r') as f:
        config = yaml.safe_load(f)
        model_conf = config['model']
        data_conf = config['dataset']
        train_conf = config['train']
    return model_conf, data_conf, train_conf


def train(train_config, model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(int(train_config['epoch'])):
        for i_batch, img in enumerate(tqdm(dataloader)):
            x, y = img[0], img[1]
            optimizer.zero_grad()
            
            pre = model(x)
            loss = criterion(pre, y)
            loss.backward()
            optimizer.step()


def main(args):
    model_config, data_config, train_config= load_config(args)
    imgloader = build_dataset(data_config)
    model = build_model(model_config)
    device = torch.device('cpu')
    model.to(device)
    train(train_config, model, imgloader)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_conf', type = str, default='./config/train_conf.yaml')
    args = parser.parse_args()
    main(args)