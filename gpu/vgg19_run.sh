#!bin/bash

VGG19_JOB1="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 1 --load_model True"
VGG19_JOB2="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 2"
VGG19_JOB3="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 4"
VGG19_JOB4="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 8"
VGG19_JOB5="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 16"
VGG19_JOB6="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 32"
VGG19_JOB7="/root/aws-elastic-inference/gpu/vgg19/vgg19_trt.py --batchsize 64"


echo "VGG19"
python3 $VGG19_JOB1 
sleep 2
python3 $VGG19_JOB2 
sleep 2
python3 $VGG19_JOB3 
sleep 2
python3 $VGG19_JOB4 
sleep 2
python3 $VGG19_JOB5 
sleep 2
python3 $VGG19_JOB6 
sleep 2
python3 $VGG19_JOB7 
sleep 2