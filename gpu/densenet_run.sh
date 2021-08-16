#!bin/bash

DENSENET121_JOB1="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 1 --load_model True"
DENSENET121_JOB2="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 2"
DENSENET121_JOB3="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 4"
DENSENET121_JOB4="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 8"
DENSENET121_JOB5="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 16"
DENSENET121_JOB6="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 32"
DENSENET121_JOB7="/root/aws-elastic-inference/gpu/densenet/densenet121_trt.py --batchsize 64"

DENSENET169_JOB1="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 1 --load_model True"
DENSENET169_JOB2="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 2"
DENSENET169_JOB3="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 4"
DENSENET169_JOB4="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 8"
DENSENET169_JOB5="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 16"
DENSENET169_JOB6="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 32"
DENSENET169_JOB7="/root/aws-elastic-inference/gpu/densenet/densenet169_trt.py --batchsize 64"

DENSENET201_JOB1="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 1 --load_model True"
DENSENET201_JOB2="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 2"
DENSENET201_JOB3="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 4"
DENSENET201_JOB4="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 8"
DENSENET201_JOB5="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 16"
DENSENET201_JOB6="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 32"
DENSENET201_JOB7="/root/aws-elastic-inference/gpu/densenet/densenet201_trt.py --batchsize 64"


echo "Densenet121"
python3 $DENSENET121_JOB1 
sleep 2
python3 $DENSENET121_JOB2
sleep 2
python3 $DENSENET121_JOB3 
sleep 2
python3 $DENSENET121_JOB4 
sleep 2
python3 $DENSENET121_JOB5 
sleep 2
python3 $DENSENET121_JOB6 
sleep 2
python3 $DENSENET121_JOB7 
sleep 2

echo "Densenet169"
python3 $DENSENET169_JOB1 
sleep 2
python3 $DENSENET169_JOB2
sleep 2
python3 $DENSENET169_JOB3
sleep 2
python3 $DENSENET169_JOB4 
sleep 2
python3 $DENSENET169_JOB5 
sleep 2
python3 $DENSENET169_JOB6 
sleep 2
python3 $DENSENET169_JOB7 
sleep 2

echo "Densenet201"
python3 $DENSENET201_JOB1 
sleep 2
python3 $DENSENET201_JOB2
sleep 2
python3 $DENSENET201_JOB3
sleep 2
python3 $DENSENET201_JOB4
sleep 2
python3 $DENSENET201_JOB5 
sleep 2
python3 $DENSENET201_JOB6 
sleep 2
python3 $DENSENET201_JOB7 
sleep 2
