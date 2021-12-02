# cross compile CPU -> GPU 
import os
import numpy as np
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=64,type=int)
parser.add_argument('--arch',default='sm_75',type=str)

args = parser.parse_args()
model = args.model
batch_size = args.batchsize
arch=args.arch


dtype = 'float32'
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  
# target = tvm.target.Target("cuda -arch=sm_75")  # or
# target = tvm.target.cuda(model="nvidia", arch="sm_75")
# print(target.arch)
target = tvm.target.Target(f'cuda -model=nvidia -arch={arch}')
# target = tvm.target.cuda()
set_cuda_target_arch('sm_75')


data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
 
import time
mod, params = relay.testing.resnet.get_workload(
            num_layers=50, batch_size=batch_size, image_shape=image_shape)

build_model = time.time()
opt_level = 3
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)
print('build_model time', (((time.time() - build_model) ) * 1000),"ms")


from tvm.contrib import utils
lib.export_library(f"./{model}_{batch_size}_{arch}_deploy_lib.tar")
