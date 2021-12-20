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
args = parser.parse_args()
model_name = args.model
# model_name = "resnet50"
batch_size = 16
size=224

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


target = "llvm"
ctx = tvm.cpu()

##### TVM compile style 1 
# print("-"*50)
# print("Compile style 1 : vm compile ")
# print("-"*50)
# with tvm.transform.PassContext(opt_level=3):
#     exe = relay.vm.compile(mod,target=target, params=params)

# vm = VirtualMachine(exe,ctx)
# _out = vm.invoke("main",data)
# # print(_out)

# input_data = tvm.nd.array(data)
# input_list = [input_data]
# warm_iterations=10
# measurements=5

# for i in range(warm_iterations):    # warm up
#     vm.run(input_list)

# start_time = time.time()
# for i in range(measurements):
#     vm.run(input_list)
# end_time = time.time()
# tvm_time = end_time - start_time
# print("VM runtime time elapsed", tvm_time/measurements)
# print("\n")

##### TVM compile style 
print("-"*50)
print("Compile style : create_executor vm ")
print("-"*50)
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor("vm", mod, tvm.cpu(0), target)

# tasks = autotvm.task.extract_from_program(
#     mod["main"], target=target, params=params)
# )

executor.evaluate()(data,**params)

input_data = tvm.nd.array(data)
input_list = [input_data]
warm_iterations=10
measurements=5

input_list = [input_data]
for i in range(warm_iterations):    # warm up
    executor.evaluate()(input_data, **params)

start_time = time.time()
for i in range(measurements):
    executor.evaluate()(input_data, **params)
end_time = time.time()
tvm_time = end_time - start_time
print(f"VM {model_name} runtime time elapsed", tvm_time/measurements,"s")
print("\n")
