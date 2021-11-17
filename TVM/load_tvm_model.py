import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

# Define Neural Network 
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

#run general 
# create random input
dev = tvm.cpu(0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# load the module 
loaded_lib = tvm.runtime.load_module('/home/ubuntu/test/model/')
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

