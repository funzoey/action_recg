from unicodedata import name
import torch
import model.Resnet as rs
import model.dense as dense
def build_model(args):
    CNN = args['CNN']
    if  CNN['name']== '':
        print('Wrong parsing')
    
    elif CNN['name'] == 'resnet50':
        return rs.resnet50(int(CNN['class_num']))

    elif CNN['name'] == 'dense161':
        return dense.densenet161(num_classes = int(CNN['class_num']))