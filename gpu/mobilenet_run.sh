#!bin/bash
MOBILENET_JOB1="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 1 --load_model True "
MOBILENET_JOB2="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 2"
MOBILENET_JOB3="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 4"
MOBILENET_JOB4="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 8"
MOBILENET_JOB5="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 16"
MOBILENET_JOB6="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 32"
MOBILENET_JOB7="/root/aws-elastic-inference/gpu/mobilenet/mobilenet_trt.py --batchsize 64"

MOBILENETv2_JOB1="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 1 --load_model True "
MOBILENETv2_JOB2="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 2"
MOBILENETv2_JOB3="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 4"
MOBILENETv2_JOB4="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 8"
MOBILENETv2_JOB5="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 16"
MOBILENETv2_JOB6="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 32"
MOBILENETv2_JOB7="/root/aws-elastic-inference/gpu/mobilenet/mobilenetv2_trt.py --batchsize 64"

echo "MOBILENET"
python3 $MOBILENET_JOB1 
sleep 2
python3 $MOBILENET_JOB2 
sleep 2
python3 $MOBILENET_JOB3 
sleep 2
python3 $MOBILENET_JOB4 
sleep 2
python3 $MOBILENET_JOB5 
sleep 2
python3 $MOBILENET_JOB6 
sleep 2
python3 $MOBILENET_JOB7 
sleep 2

echo "MOBILENETv2"
python3 $MOBILENETv2_JOB1 
sleep 2
python3 $MOBILENETv2_JOB2
sleep 2
python3 $MOBILENETv2_JOB3 
sleep 2
python3 $MOBILENETv2_JOB4 
sleep 2
python3 $MOBILENETv2_JOB5 
sleep 2
python3 $MOBILENETv2_JOB6 
sleep 2
python3 $MOBILENETv2_JOB7 
sleep 2
