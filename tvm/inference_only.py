import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1)
args = parser.parse_args()
batch_size = args.batchsize
# Define Neural Network 

model = 'resnet50'
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

#run general 
# create random input
dev = tvm.cuda(0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
input_data = tvm.nd.array(data)

# load the module 
end_to_end = time.time()

loaded_lib = tvm.runtime.load_module('/home/ubuntu/test/model/t2_T4_resnet50_deploy_lib.tar')

mm = graph_executor.GraphModule(loaded_lib["default"](dev))
mm.run(data=input_data)
out_deploy = mm.get_output(0).numpy()
print((time.time()-end_to_end)*1000,"ms")

e = mm.module.time_evaluator("run", dev, number=20, repeat=10)
t = e(input_data).results
t = np.array(t) * 1000

print('{} (batch={}): {} ms'.format(model, batch_size, t.mean()))

# tvm_result = []
# for i in range(20):
#     start = time.time()
#     mm.run()
#     tvm_out = mm.get_output(0)
#     tvm_result.append(((time.time() - start) * 1000))
#     print('tvm compile on CPU :', i, 'th time', (time.time() - start) * 1000,"ms")
