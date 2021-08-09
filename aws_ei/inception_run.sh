#!bin/bash

echo "INCEPTION_V3"
INCEPTIONv3_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 1 --load_model True"
INCEPTIONv3_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 2"
INCEPTIONv3_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 4"
INCEPTIONv3_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 8"
INCEPTIONv3_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 16"
INCEPTIONv3_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 32"
INCEPTIONv3_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inceptionV3_both.py --batchsize 64"

python3 $INCEPTIONv3_JOB1
sleep 2
python3 $INCEPTIONv3_JOB2
sleep 2
python3 $INCEPTIONv3_JOB3 
sleep 2
python3 $INCEPTIONv3_JOB4
sleep 2
python3 $INCEPTIONv3_JOB5 
sleep 2
python3 $INCEPTIONv3_JOB6 
sleep 2
python3 $INCEPTIONv3_JOB7 
sleep 2

echo "INCEPTION_RESNET_V2"
INCEPTIONresnetV2_JOB1="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 1 --load_model True"
INCEPTIONresnetV2_JOB2="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 2"
INCEPTIONresnetV2_JOB3="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 4"
INCEPTIONresnetV2_JOB4="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 8"
INCEPTIONresnetV2_JOB5="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 16"
INCEPTIONresnetV2_JOB6="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 32"
INCEPTIONresnetV2_JOB7="/home/ubuntu/aws-elastic-inference/aws_ei/inception/inception_resnet_v2_both.py --batchsize 64"

python3 $INCEPTIONresnetV2_JOB1
sleep 2
python3 $INCEPTIONresnetV2_JOB2
sleep 2
python3 $INCEPTIONresnetV2_JOB3 
sleep 2
python3 $INCEPTIONresnetV2_JOB4
sleep 2
python3 $INCEPTIONresnetV2_JOB5 
sleep 2
python3 $INCEPTIONresnetV2_JOB6 
sleep 2
python3 $INCEPTIONresnetV2_JOB7 
sleep 2
