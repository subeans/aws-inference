#!bin/bash

XCEPTION_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 1 --load_model True"
XCEPTION_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 2"
XCEPTION_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 4"
XCEPTION_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 8"
XCEPTION_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 16"
XCEPTION_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 32"
XCEPTION_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/xception/xception_both.py --batchsize 64"

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
