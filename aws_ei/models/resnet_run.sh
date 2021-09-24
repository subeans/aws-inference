#!bin/bash

RESNET50_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 1 --load_model True"
RESNET50_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 2"
RESNET50_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 4"
RESNET50_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 8"
RESNET50_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 16"
RESNET50_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 32"
RESNET50_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet50_both.py --batchsize 64"

echo "RESNET50"
python3 $RESNET50_JOB1 
sleep 2
python3 $RESNET50_JOB2 
sleep 2
python3 $RESNET50_JOB3 
sleep 2
python3 $RESNET50_JOB4 
sleep 2
python3 $RESNET50_JOB5 
sleep 2
python3 $RESNET50_JOB6 
sleep 2
python3 $RESNET50_JOB7 
sleep 2

RESNET101_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 1 --load_model True"
RESNET101_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 2"
RESNET101_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 4"
RESNET101_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 8"
RESNET101_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 16"
RESNET101_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 32"
RESNET101_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet101_both.py --batchsize 64"

echo "RESNET101"
python3 $RESNET101_JOB1
sleep 2
python3 $RESNET101_JOB2 
sleep 2
python3 $RESNET101_JOB3 
sleep 2
python3 $RESNET101_JOB4 
sleep 2
python3 $RESNET101_JOB5 
sleep 2
python3 $RESNET101_JOB6 
sleep 2
python3 $RESNET101_JOB7 
sleep 2

RESNET152_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 1 --load_model True"
RESNET152_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 2"
RESNET152_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 4"
RESNET152_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 8"
RESNET152_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 16"
RESNET152_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 32"
RESNET152_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/resnet/resnet152_both.py --batchsize 64"

echo "RESNET152"
python3 $RESNET152_JOB1
sleep 2
python3 $RESNET152_JOB2
sleep 2
python3 $RESNET152_JOB3 
sleep 2
python3 $RESNET152_JOB4 
sleep 2
python3 $RESNET152_JOB5 
sleep 2
python3 $RESNET152_JOB6 
sleep 2
python3 $RESNET152_JOB7 
sleep 2
