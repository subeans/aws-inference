import sys
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from time import time

from tvm.relay import testing
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

def get_input(input_shape):
    im_fname = download_testdata(
        'https://github.com/dmlc/web-data/blob/master/' +
        'gluoncv/detection/street_small.jpg?raw=true',
        'street_small.jpg',
        module='data')
    x, img = gcv.data.transforms.presets.ssd.load_test(im_fname,
                                                       short=input_shape[2])
    return x.asnumpy().astype('float32')


def compile_model(input_shape, target):
    shape_dict = {'data': input_shape}
    # tvm_net, tvm_params = relay.frontend.from_mxnet(net, shape_dict)
    mod, params = relay.testing.resnet.get_workload(
    num_layers=50, batch_size=batch_size, image_shape=image_shape
        )
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
    return graph, lib, params


def run(graph, lib, params, x, ctx):
    m = graph_runtime.create(graph, lib, ctx)
    # tvm_input = tvm.nd.array(x, ctx=ctx)
    m.set_input('data', x)
    m.set_input(**params)
    return m


if __name__ == "__main__":

    # target = tvm.target.create('llvm -mcpu=haswell')
    target='llvm'
    tvm_ctx = tvm.cpu()
    # target = tvm.target.cuda('1080ti')
    # tvm_ctx = tvm.gpu()

    model_name = 'ResNet50'
    input_shape = (1, 3, 224, 224)
    tvm_result = list()

    # get input
    # x = get_input(input_shape)
    x = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    x = tvm.nd.array(x.astype('float32'))

    # compile model
    graph, lib, params = compile_model( input_shape, target)
    m = run(graph, lib, params, x, tvm_ctx)

    print('start', model_name, 'speed benchmark')

    for idx in range(10):
        # warm up
        m.run()
        tvm_out = m.get_output(0)
        # speed benchmark
        start = time()
        for _ in range(100):
            m.run()
            tvm_out = m.get_output(0)
        tvm_result.append(((time() - start) / 100) * 1000)
        print(' tvm :', idx, 'time', (((time() - start) / 100) * 1000))

    print(' tvm :', model_name, np.min(tvm_result), np.mean(tvm_result),
          np.max(tvm_result))