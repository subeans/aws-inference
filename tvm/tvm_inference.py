import numpy as np
import time 
import argparse

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1)
args = parser.parse_args()
batch_size = args.batchsize

# Define Neural Network 
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

#run general 
# create random input
dev = tvm.cuda()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")


# load module time check 
load_model = time.time()
loaded_lib = tvm.runtime.load_module(f'/home/ubuntu/test/model/resnet50_{batch_size}_sm_75_deploy_lib.tar')
print('load_model time', (((time.time() - load_model) ) * 1000),"ms")


module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input("data", data)
module.run()
out_deploy = module.get_output(0).numpy()


# Print first 10 elements of output
# print(out_deploy.flatten()[0:10])

print("Time check using TVM module")
input_data = tvm.nd.array(data)
e = module.module.time_evaluator("run", dev, number=10, repeat=1)
t = e(input_data).results
t = np.array(t) * 1000
model = 'resnet50-crosscompile'
print('{} (batch={}): {} ms'.format(model, batch_size, t.mean()))

print("Time check using time.time ")
tvm_result = []
for i in range(10):
    start = time.time()
    module.run(data=data)
    tvm_out = module.get_output(0)
    tvm_result.append((time.time()-start)*1000)
    print(' tvm ', i, 'th time :', (((time.time() - start) ) * 1000),"ms")

print('tvm average :',np.mean(tvm_result),"ms")
print('tvm average except first batch :',np.mean(tvm_result[1:]),"ms")
