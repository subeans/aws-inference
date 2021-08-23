import os
import time
import shutil
import json
import time
import numpy as np
import requests
from functools import partial
import argparse

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
from tensorflow.keras.preprocessing import image
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version

print(f"TensorRT version: {get_linked_tensorrt_version()}")
print(f"TensorFlow version: {tf.__version__}")

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50',type=str)
parser.add_argument('--batchsize',default=8 , type=int)
parser.add_argument('--precision',default='fp32', type=str)
parser.add_argument('--size',default=224,type=int)
args = parser.parse_args()
model = args.model 
batch_size = args.batchsize
precision = args.precision
size=args.size

models = {
    'xception':xception,
    'vgg16':vgg16,
    'vgg19':vgg19,
    'resnet50':resnet50,
    'resnet101':resnet,
    'resnet152':resnet,
    'resnet50_v2':resnet_v2,
    'resnet101_v2':resnet_v2,
    'resnet152_v2':resnet_v2,
    'inception_v3':inception_v3,
    'inception_resnet_v2':inception_resnet_v2,
    'mobilenet':mobilenet,
    'densenet121':densenet,
    'densenet169':densenet,
    'densenet201':densenet,
    'nasnetlarge':nasnet,
    'nasnetmobile':nasnet,
    'mobilenet_v2':mobilenet_v2
}

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                  'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    label_text = tf.cast(obj['image/class/text'], tf.string)   
    return imgdata, label, label_text

def val_preprocessing(record):
    imgdata, label, label_text = deserialize_image_record(record)
    label -= 1
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, size, size)
    
    image = models[model].preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = '/root/datasets/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    if use_cache:
        shutil.rmtree('tfdatacache', ignore_errors=True)
        os.mkdir('tfdatacache')
        dataset = dataset.cache(f'./tfdatacache/imagenet_val')
    
    return dataset

####tensorRT compile

def build_fn(batch_size, dataset):
    for i, (build_image, _, _) in enumerate(dataset):
        if i > 1:
            break
        yield (build_image,)

def calibrate_fn(n_calib, batch_size, dataset):
    for i, (calib_image, _, _) in enumerate(dataset):
        if i > n_calib // batch_size:
            break
        yield (calib_image,)

def build_tensorrt_engine(precision, batch_size, dataset,model):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    input_saved_model = f'{model}_saved_model'
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision.upper(),
                                                                   max_workspace_size_bytes=(1<<32),
                                                                   maximum_cached_engines=2)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model,
                                        conversion_params=conversion_params)
    
    if precision.lower() == 'int8':
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batch_size, 
                                                       dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))
    else:
        converter.convert()
        
    trt_compiled_model_dir = f'{model}_trt_saved_models/{model}_{precision}_{batch_size}'
    shutil.rmtree(trt_compiled_model_dir, ignore_errors=True)

    converter.build(input_fn=partial(build_fn, batch_size, dataset))
    converter.save(output_saved_model_dir=trt_compiled_model_dir)
    print(f'\nOptimized for {precision} and batch size {batch_size}, directory:{trt_compiled_model_dir}\n')
    return trt_compiled_model_dir


dataset = get_dataset(batch_size)  

trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset,model)



