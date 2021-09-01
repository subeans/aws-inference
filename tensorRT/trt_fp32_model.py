import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
import logging
import numpy as np

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
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

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saved_model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=64,type=int)
args = parser.parse_args()
model = args.saved_model
BATCH_SIZE=args.batchsize
SAVED_MODEL_DIR=f'{model}_saved_model'


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



def get_files(data_dir, filename_pattern):
    if data_dir == None:
        return []
    files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with '
                         'pattern "{}"'.format(data_dir, filename_pattern))
    return files

VALIDATION_DATA_DIR = "/data"
validation_files = get_files(VALIDATION_DATA_DIR, 'validation*')
print('There are %d validation files. \n%s\n%s\n...'%(len(validation_files), validation_files[0], validation_files[-1]))



def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.io.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.io.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.io.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text


from tensorflow.keras.preprocessing import image
#from preprocessing import vgg_preprocess as vgg_preprocessing
def preprocess(record):
    # Parse TFRecord
    imgdata, label, bbox, text = deserialize_image_record(record)
    #label -= 1 # Change to 0-based if not using background class
    try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
    except: image = tf.image.decode_png(imgdata, channels=3)
        
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
    image = tf.image.resize_with_crop_or_pad(image,224 , 224)

    image = models[model].preprocess_input(image)
    #image = vgg_preprocessing(image, 224, 224)
    
    return image, label

INPUT_TENSOR = 'input_tensor:0'
OUTPUT_TENSOR = 'softmax_tensor:0'

def benchmark_saved_model(SAVED_MODEL_DIR, BATCH_SIZE=64):
    # load saved model
    saved_model_loaded = tf.saved_model.load(SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    # prepare dataset iterator
    dataset = tf.data.TFRecordDataset(validation_files)
    dataset = dataset.map(map_func=preprocess, num_parallel_calls=20)
    dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print('Warming up for 50 batches...')
    cnt = 0
    for x, y in dataset:
        labeling = infer(x)
        cnt += 1
        if cnt == 50:
            break

    print('Benchmarking inference engine...')
    num_hits = 0
    num_predict = 0
    start_time = time.time()
    for x, y in dataset:
        labeling = infer(x)
        preds = np.array(list(np.argmax(labeling['probs'],axis=1)))
        num_hits += np.sum(preds == y)
        num_predict += preds.shape[0]

    print('Accuracy: %.2f%%'%(100*num_hits/num_predict))
    print('Inference speed: %.2f samples/s'%(num_predict/(time.time()-start_time)))

FP32_SAVED_MODEL_DIR = SAVED_MODEL_DIR+"_TFTRT_FP32"

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP32)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR,
conversion_params=conversion_params)
converter.convert()

converter.save(FP32_SAVED_MODEL_DIR)


benchmark_saved_model(FP32_SAVED_MODEL_DIR, BATCH_SIZE=BATCH_SIZE)