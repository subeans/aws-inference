import os
import time
import numpy as np
import pandas as pd
import shutil
import requests
from functools import partial

import tensorflow as tf
#import tensorflow.keras as keras
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants, signature_constants


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
parser.add_argument('--batchsize',default=64,type=int)
parser.add_argument('--precision',default='FP32',type=str)
parser.add_argument('--load',default=False,type=bool)
parser.add_argument('--gpu',default=False,type=bool)
parser.add_argument('--engines',default=2,type=int)
args = parser.parse_args()
load_model = args.model
batch_size = args.batchsize
precision = args.precision
load=args.load
run_gpu=args.gpu
num_engine=args.engines


def load_save_model(load_model , saved_model_dir = 'resnet50_saved_model'):

    model = models_detail[load_model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    model.save(saved_model_dir, save_format='tf')

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
    
    image = models[load_model].preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size,datafolder, use_cache=False):
    data_dir = f'/home/ubuntu/{datafolder}/*'
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



def predict_GPU(batch_size,saved_model_dir):
    display_every = 5000
    display_threshold = display_every

    pred_labels = []
    actual_labels = []
    iter_times = []

    dataset = get_dataset(batch_size,'datasets')  


    walltime_start = time.time()
    model = tf.keras.models.load_model(saved_model_dir)
    
    N=0
    warm_up=50
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        N+=1
        if i==0:
            for w in range(warm_up):
                _ = model(validation_ds)
        start_time = time.time()
        #pred_prob_keras = model(validation_ds)
        with tf.device("/device:GPU:0"):
            pred_prob_keras = model.predict(validation_ds)
        iter_times.append(time.time() - start_time)
        
        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(np.argmax(pred_prob_keras, axis=1)))
        
        if i*batch_size >= display_threshold:
            print(f'Images {i*batch_size}/50000. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every

    iter_times = np.array(iter_times)
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    print('Throughput: {:.0f} images/s'.format(N * batch_size / sum(iter_times)))

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
    print(results)


saved_model_dir = f'{load_model}_saved_model'

if load :
    load_save_model(load_model,saved_model_dir)

predict_GPU(batch_size,saved_model_dir)
