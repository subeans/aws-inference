#!bin/bash
MOBILENET_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 1 --load_model True "
MOBILENET_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 2"
MOBILENET_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 4"
MOBILENET_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 8"
MOBILENET_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 16"
MOBILENET_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 32"
MOBILENET_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenet_both.py --batchsize 64"

MOBILENETv2_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 1 --load_model True "
MOBILENETv2_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 2"
MOBILENETv2_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 4"
MOBILENETv2_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 8"
MOBILENETv2_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 16"
MOBILENETv2_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 32"
MOBILENETv2_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/mobilenet/mobilenetV2_both.py --batchsize 64"

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
