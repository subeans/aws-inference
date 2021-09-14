import os
import shutil

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ( 
    vgg16,
    vgg19,
    resnet,
    resnet50,
    resnet_v2,
    inception_v3,
    inception_resnet_v2,
    mobilenet,
    mobilenet_v2,
    densenet,
    nasnet,
    xception,
)

models_detail = {
    'xception':xception.Xception(weights='imagenet',include_top=False),
    'vgg16':vgg16.VGG16(weights='imagenet'),
    'vgg19':vgg19.VGG19(weights='imagenet'),
    'resnet50':resnet50.ResNet50(weights='imagenet'),
    'resnet101':resnet.ResNet101(weights='imagenet'),
    'resnet152':resnet.ResNet152(weights='imagenet'),
    'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
    'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
    'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
    'inception_v3':inception_v3.InceptionV3(weights='imagenet',include_top=False),
    'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
    'densenet121':densenet.DenseNet121(weights='imagenet'),
    'densenet169':densenet.DenseNet169(weights='imagenet'),
    'densenet201':densenet.DenseNet201(weights='imagenet'),
    'nasnetlarge':nasnet.NASNetLarge(weights='imagenet'),
    'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
    'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet')
}

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
args = parser.parse_args()
load_model = args.model

saved_model_dir = f'{load_model}_saved_model'

def load_save_model(load_model,saved_model_dir = 'resnet50_saved_model'):
    print(models_detail[load_model])
    model = models_detail[load_model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    try:
        model.save(saved_model_dir, include_optimizer=False, save_format='tf')
        print(saved_model_dir," : complete load ")
    except:
        print("NOT saved")

if load_model : 
    load_save_model(load_model,saved_model_dir)

