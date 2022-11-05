from unicodedata import name
import torch
import model.Resnet as rs
import model.dense_clasifier as dense_clasifier
from torchvision.models import vision_transformer
def build_model(args):
    CNN = args['CNN']
    if  CNN['name']== '':
        print('Wrong parsing')
    
    elif CNN['name'] == 'resnet50':
        return rs.resnet50(int(CNN['class_num']))

    elif CNN['name'] == 'dense161':
        return dense_clasifier.densenet161(num_classes = int(CNN['class_num']))

    elif CNN['name'] == 'vitb16':
        return vision_transformer.VisionTransformer(image_size=224,
                                                    patch_size=16,
                                                    num_layers=12,
                                                    num_heads=12,
                                                    hidden_dim=768,
                                                    mlp_dim=3072,
                                                    num_classes=CNN['class_num'])