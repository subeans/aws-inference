#!bin/bash

VGG16_JOB1="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 1 --load_model True"
VGG16_JOB2="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 2"
VGG16_JOB3="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 4"
VGG16_JOB4="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 8"
VGG16_JOB5="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 16"
VGG16_JOB6="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 32"
VGG16_JOB7="/root/aws-elastic-inference/gpu/vgg16/vgg16_trt.py --batchsize 64"

python3 $VGG16_JOB1 
sleep 2
python3 $VGG16_JOB2 
sleep 2
python3 $VGG16_JOB3 
sleep 2
python3 $VGG16_JOB4 
sleep 2
python3 $VGG16_JOB5 
sleep 2
python3 $VGG16_JOB6 
sleep 2
python3 $VGG16_JOB7 
sleep 2
