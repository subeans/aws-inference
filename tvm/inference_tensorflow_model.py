import tensorflow as tf
from tvm import relay
from PIL import Image
import os
import numpy as np 
import tvm 
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor


batch_size = 1
size=224
data,image_shape = make_dataset(batch_size,size)
shape_dict = {"DecodeJpeg/contents": data.shape}

def make_dataset(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

  
with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["input_1:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)


mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
print("Tensorflow protobuf imported to relay frontend.")

target = "llvm"
ctx = tvm.cpu()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod,target=target, params=params)

# graph create 
module = graph_executor.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_deploy = module.get_output(0).numpy()

lib.export_library(f"./resnet50_{batch_size}_{target}_deploy_lib.tar")

data_tvm = tvm.nd.array(data.astype('float32'))
e = module.module.time_evaluator("run", ctx, number=5, repeat=1)
t = e(data_tvm).results
t = np.array(t) * 1000

print('{} (batch={}): {} ms'.format('resnet50', batch_size, t.mean()))

