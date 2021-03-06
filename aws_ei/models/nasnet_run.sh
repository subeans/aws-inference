#!bin/bash

NASNETLARGE_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 1 --load_model True"
NASNETLARGE_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 2"
NASNETLARGE_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 4"
NASNETLARGE_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 8"
NASNETLARGE_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 16"
NASNETLARGE_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 32"
NASNETLARGE_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetlarge_both.py --batchsize 64"

NASNETMOBILE_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 1 --load_model True"
NASNETMOBILE_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 2"
NASNETMOBILE_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 4"
NASNETMOBILE_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 8"
NASNETMOBILE_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 16"
NASNETMOBILE_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 32"
NASNETMOBILE_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/nasnet/nasnetmobile_both.py --batchsize 64"


echo "NasNet_Large"
python3 $NASNETLARGE_JOB1 
sleep 2
python3 $NASNETLARGE_JOB2
sleep 2
python3 $NASNETLARGE_JOB3 
sleep 2
python3 $NASNETLARGE_JOB4 
sleep 2
python3 $NASNETLARGE_JOB5 
sleep 2
python3 $NASNETLARGE_JOB6 
sleep 2
python3 $NASNETLARGE_JOB7 
sleep 2

echo "NasNet_Mobile"
python3 $NASNETMOBILE_JOB1 
sleep 2
python3 $NASNETMOBILE_JOB2
sleep 2
python3 $NASNETMOBILE_JOB3 
sleep 2
python3 $NASNETMOBILE_JOB4 
sleep 2
python3 $NASNETMOBILE_JOB5 
sleep 2
python3 $NASNETMOBILE_JOB6 
sleep 2
python3 $NASNETMOBILE_JOB7 
sleep 2
