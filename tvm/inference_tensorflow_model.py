import itertools
import tensorflow as tf
from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
import tvm.relay.testing.tf as tf_testing

import tvm.testing 
from tvm.runtime.vm import VirtualMachine
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--imgsize',default=224 , type=int)


args = parser.parse_args()

model_name = args.model
batch_size = args.batchsize
size = args.imgsize
# model_name = "resnet50"
# size=224

def make_dataset(batch_size,size):
    image_shape = (size, size,3)
    # image_shape = (3,size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    # print("-" * 50)
    # print("Frozen model layers: ")
    # layers = [op.name for op in import_graph.get_operations()]
    # if print_graph == True:
    #     for layer in layers:
    #         print(layer)
    # print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


# saved model -> frozen model 변환한 모델 호출하여 wrap 하기 
load_model = time.time()
with tf.io.gfile.GFile(f"./frozen_models/frozen_{model_name}.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        try : 
            with tf.compat.v1.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
        except:
            pass

frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["input_1:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

# test dataset 생성 
data,image_shape = make_dataset(batch_size,size)

print(data.shape)
shape_dict = {"DecodeJpeg/contents": data.shape}

##### Convert tensorflow model 
mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
print("Tensorflow protobuf imported to relay frontend.")
print("-"*10,"Load frozen model",time.time()-load_model,"s","-"*10)

target = "llvm"
ctx = tvm.cpu()


##### TVM compile style 
print("-"*10,"Compile style : create_executor vm ","-"*10)
build_time = time.time()
with tvm.transform.PassContext(opt_level=3):
    # executor = relay.build_module.create_executor("vm", mod, tvm.cpu(0), target)
    executor = relay.vm.compile(mod, target=target, params=params)

print("-"*10,"Build latency : ",time.time()-build_time,"s","-"*10)

# executor.evaluate()(data,**params)
vm = VirtualMachine(executor,ctx)
_out = vm.invoke("main",data)


input_data = tvm.nd.array(data)
warm_iterations=3
measurements=5

for i in range(warm_iterations):    # warm up
    vm.run(input_data)

iter_times = []
for i in range(measurements):
    start_time = time.time()
    vm.run(input_data)
    print(f"VM {model_name}-{batch_size} inference latency : ",(time.time()-start_time)*1000,"ms")
    iter_times.append(time.time() - start_time)

print(f"VM {model_name}-{batch_size} inference mean latency ",np.mean(iter_times) * 1000 ,"ms")
print("\n")
