#!bin/bash

XCEPTION_JOB1="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 1 --load_model True"
XCEPTION_JOB2="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 2"
XCEPTION_JOB3="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 4"
XCEPTION_JOB4="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 8"
XCEPTION_JOB5="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 16"
XCEPTION_JOB6="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 32"
XCEPTION_JOB7="/root/aws-elastic-inference/gpu/xception/xception_trt.py --batchsize 64"

echo "XCEPTION"

python3 $XCEPTION_JOB1
sleep 2
python3 $XCEPTION_JOB2 
sleep 2
python3 $XCEPTION_JOB3 
sleep 2
python3 $XCEPTION_JOB4 
sleep 2
python3 $XCEPTION_JOB5 
sleep 2
python3 $XCEPTION_JOB6 
sleep 2
python3 $XCEPTION_JOB7 
sleep 2
