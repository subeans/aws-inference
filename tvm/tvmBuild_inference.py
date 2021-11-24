import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
import time

import pickle
from tvm.contrib import graph_runtime
from tvm.relay.testing import resnet
from tvm.relay.testing import mobilenet
from tvm.relay.testing import inception_v3



def test_resnet():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    data_tvm = tvm.nd.array(data.astype('float32'))

    target = tvm.target.cuda('Tesla_T4')
    ctx = tvm.cuda()

    mod, params = relay.testing.resnet.get_workload(
        num_layers=50, batch_size=batch_size, image_shape=image_shape
    )

    end_to_end = time.time()

    # build 
    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    # graph create 
    module = graph_runtime.create(graph, lib, ctx)

    # set data 
    # module.set_input("data", data)
    # module.set_input(**params)

    module.run(data = data)
    out = module.get_output(0).asnumpy()

    print((time.time()-end_to_end)*1000,"ms")

    e = module.module.time_evaluator("run", ctx, number=20, repeat=10)
    t = e(data_tvm).results
    t = np.array(t) * 1000

    print('{} (batch={}): {} ms'.format('resnet50', batch_size, t.mean()))

def test_mobilenet():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    mod, params = mobilenet.get_workload()
    target = tvm.target.cuda('Tesla_T4')
    ctx = tvm.cuda()

    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()


def test_inception():
    batch_size = 1
    image_shape = (3, 299, 299)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    mod, params = inception_v3.get_workload()
    target = tvm.target.cuda('Tesla_T4')
    ctx = tvm.cuda()

    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()


if __name__ == "__main__":
    test_resnet()
