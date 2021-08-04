#!bin/bash
MOBILENET_JOB1="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 1 --load_model True "
MOBILENET_JOB2="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 2"
MOBILENET_JOB3="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 4"
MOBILENET_JOB4="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 8"
MOBILENET_JOB5="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 16"
MOBILENET_JOB6="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 32"
MOBILENET_JOB7="/home/ubuntu/aws-elastic-inference/gpu/mobilenet/mobilenet_gpu.py --batchsize 64"

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
