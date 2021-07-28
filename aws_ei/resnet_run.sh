#!bin/bash

RESNET_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 1 --load_model True"
RESNET_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 2"
RESNET_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 4"
RESNET_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 8"
RESNET_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 16"
RESNET_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 32"
RESNET_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/resnet50/resnet50_both.py --batchsize 64"

python3 $RESNET_JOB1 
sleep 2
python3 $RESNET_JOB2 
sleep 2
python3 $RESNET_JOB3 
sleep 2
python3 $RESNET_JOB4 
sleep 2
python3 $RESNET_JOB5 
sleep 2
python3 $RESNET_JOB6 
sleep 2
python3 $RESNET_JOB7 
sleep 2
