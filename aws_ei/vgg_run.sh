#!bin/bash

VGG16_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 1 --load_model True"
VGG16_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 2"
VGG16_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 4"
VGG16_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 8"
VGG16_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 16"
VGG16_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 32"
VGG16_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg16_both.py --batchsize 64"

VGG19_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 1 --load_model True"
VGG19_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 2"
VGG19_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 4"
VGG19_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 8"
VGG19_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 16"
VGG19_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 32"
VGG19_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/vgg/vgg19_both.py --batchsize 64"

echo "VGG16"
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
