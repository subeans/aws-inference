import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from ei_for_tf.python.predictor.ei_predictor import EIPredictor
import numpy as np
import pandas as pd
import shutil
import requests
import time
import json
import os
import boto3
import argparse

print(tf.__version__) 

# https://github.com/tensorflow/tensorflow/issues/29931
temp = tf.zeros([8, 224, 224, 3])
_ = tf.keras.applications.vgg16.preprocess_input(temp)

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--load_model',default=False , type=bool)
args = parser.parse_args()
batch_size = args.batchsize
load_model = args.load_model

ei_client = boto3.client('elastic-inference')
print(json.dumps(ei_client.describe_accelerators()['acceleratorSet'], indent=1))

def load_save_model(saved_model_dir = 'vgg16_saved_model'):
    model = VGG16(weights='imagenet')
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
    
    image = tf.keras.applications.vgg16.preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ubuntu/datasets/*'
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

saved_model_dir = 'vgg16_saved_model' 
if load_model : 
    load_save_model(saved_model_dir)
    
print('\n=======================================================')
print(f'Benchmark results for CPU Keras, batch size: {batch_size}')
print('=======================================================\n')

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

results = pd.DataFrame(columns = [f'keras_cpu'])
results.loc['instance_type']           = [requests.get('http://169.254.169.254/latest/meta-data/instance-type').text]
results.loc['accelerator']             = ['NA']
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
print(results.T)

