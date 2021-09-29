import os
import time
import shutil
import json
import time
import pandas as pd
import numpy as np
import requests
from functools import partial
import argparse

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
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
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version

print(f"TensorRT version: {get_linked_tensorrt_version()}")
print(f"TensorFlow version: {tf.__version__}")


results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50',type=str)
parser.add_argument('--batchsize',default=8 , type=int)
parser.add_argument('--load_batchsize',default=8 , type=int)
parser.add_argument('--precision',default='fp32', type=str)
parser.add_argument('--size',default=224,type=int)
args = parser.parse_args()
model = args.model 
batch_size = args.batchsize
model_batchsize=args.load_batchsize
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

def trt_predict_benchmark(precision, batch_size, use_cache=False, display_every=100, warm_up=10):

    print('\n=======================================================')
    print(f'Benchmark results for precision: {precision}, batch size: {batch_size}')
    print('=======================================================\n')
    
    dataset = get_dataset(batch_size)
    
    # If caching is enabled, cache dataset for better i/o performance
    if use_cache:
        print('Caching dataset ...')
        start_time = time.time()
        for (img,_,_) in dataset:
            continue
        print(f'Finished caching {time.time() - start_time}')

    # LOAD 만 해서 inference 진행
    trt_compiled_model_dir = f'{model}_trt_saved_models/{model}_{precision}_{model_batchsize}'
    # trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset)

    walltime_start = time.time()

    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    initial_time = time.time()
    
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        if i==0:
            for w in range(warm_up):
                _ = model_trt(validation_ds);
                
        start_time = time.time()
        trt_results = model_trt(validation_ds);
        iter_times.append(time.time() - start_time)
        
        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(tf.argmax(trt_results['predictions'], axis=1).numpy()))
        if (i)*batch_size >= display_threshold:
            print(f'Images {(i)*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every
    
    print(f'Wall time: {time.time() - walltime_start}')

    acc_trt = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    iter_times = np.array(iter_times)
   
    results = pd.DataFrame(columns = [f'trt_{precision}_{batch_size}'])
    results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
    results.loc['user_batch_size']         = [batch_size]
    results.loc['accuracy']                = [acc_trt]
    results.loc['prediction_time']         = [np.sum(iter_times)]
    results.loc['wall_time']               = [time.time() - walltime_start]   
    results.loc['images_per_sec_mean']     = [np.mean(batch_size / iter_times)]
    results.loc['images_per_sec_std']      = [np.std(batch_size / iter_times, ddof=1)]
    results.loc['latency_mean']            = [np.mean(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_min']             = [np.min(iter_times) * 1000]
    results.loc['first_batch']             = [iter_times[0]]
    results.loc['next_batches_mean']       = [np.mean(iter_times[1:])]
    print(results.T)
   
    return results, iter_times


if results is None:
    results = pd.DataFrame()

res, it = trt_predict_benchmark(precision, batch_size)
results = res

print(results)
