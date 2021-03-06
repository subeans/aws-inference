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
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version

print(f"TensorRT version: {get_linked_tensorrt_version()}")
print(f"TensorFlow version: {tf.__version__}")

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--load_model',default=False , type=bool)
args = parser.parse_args()
batch_size = args.batchsize
load_model = args.load_model



def load_save_model(saved_model_dir = 'resnet101_saved_model'):
    model = ResNet101(weights='imagenet')
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')


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
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    
    image = tf.keras.applications.resnet.preprocess_input(image)
    
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

saved_model_dir = 'resnet101_saved_model' 
if load_model : 
    load_save_model(saved_model_dir)

model = tf.keras.models.load_model(saved_model_dir)
display_every = 5000
display_threshold = display_every

pred_labels = []
actual_labels = []
iter_times = []

# Get the tf.data.TFRecordDataset object for the ImageNet2012 validation dataset
dataset = get_dataset(batch_size)  

walltime_start = time.time()
for i, (validation_ds, batch_labels, _) in enumerate(dataset):
    start_time = time.time()
    pred_prob_keras = model(validation_ds)
    iter_times.append(time.time() - start_time)
    
    actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
    pred_labels.extend(list(np.argmax(pred_prob_keras, axis=1)))
    
    if i*batch_size >= display_threshold:
        print(f'Images {i*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
        display_threshold+=display_every

iter_times = np.array(iter_times)
acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)

results = pd.DataFrame(columns = [f'keras_gpu_{batch_size}'])
results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
results.loc['user_batch_size']         = [batch_size]
results.loc['accuracy']                = [acc_keras_gpu]
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

def build_tensorrt_engine(precision, batch_size, dataset):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision.upper(),
                                                                   max_workspace_size_bytes=(1<<32),
                                                                   maximum_cached_engines=2)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir='resnet101_saved_model',
                                        conversion_params=conversion_params)
    
    if precision.lower() == 'int8':
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batch_size, 
                                                       dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))
    else:
        converter.convert()
        
    trt_compiled_model_dir = f'resnet101_trt_saved_models/resnet101_{precision}_{batch_size}'
    shutil.rmtree(trt_compiled_model_dir, ignore_errors=True)

    converter.build(input_fn=partial(build_fn, batch_size, dataset))
    converter.save(output_saved_model_dir=trt_compiled_model_dir)
    print(f'\nOptimized for {precision} and batch size {batch_size}, directory:{trt_compiled_model_dir}\n')
    return trt_compiled_model_dir

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
    
    trt_compiled_model_dir = build_tensorrt_engine(precision, batch_size, dataset)
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    display_every = 5000
    display_threshold = display_every
    initial_time = time.time()
    
    walltime_start = time.time()
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

import itertools
bench_options = {
    'batch_size': [batch_size],
    'precision': ['fp32', 'fp16', 'int8']
}

bname, bval = zip(*bench_options.items())
blist = [dict(zip(bname, h)) for h in itertools.product(*bval)]

print('Benchmark sweep combinations:')
for b in blist:
    print(b)

iter_ds = pd.DataFrame()

if results is None:
    results = pd.DataFrame()

col_name = lambda boption: f'trt_{boption["precision"]}_{boption["batch_size"]}'

for boption in blist:
    res, it = trt_predict_benchmark(**boption)
    iter_ds = pd.concat([iter_ds, pd.DataFrame(it, columns=[col_name(boption)])], axis=1)
    results = pd.concat([results, res], axis=1)

print(results)
