import torch
import argparse
import yaml
from utils.dataset_builder import build_dataset
from utils.model_builder import build_model
from asyncio.windows_events import NULL


def load_config(args):
    with open(args.train_conf, 'r') as f:
        config = yaml.safe_load(f)
        model_conf = config['model']
        data_conf = config['dataset']

    return model_conf, data_conf



def main(args):
    model_config, data_config= load_config(args)
    imgloader = build_dataset(data_config)
    model = build_model(model_config)
    for i_batch,img in enumerate(imgloader):
        print(type(img[0]), type(img[1]))
        break

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_conf', type = str, default='./config/train_conf.yaml')
    args = parser.parse_args()
    main(args)