import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
import time

import pickle
from tvm.contrib import utils
from tvm.contrib import graph_runtime
from tvm.relay.testing import resnet
from tvm.relay.testing import mobilenet
from tvm.relay.testing import inception_v3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--imgsize',default=224 , type=int)


args = parser.parse_args()

model_name = args.model
batch_size = args.batchsize
size = args.imgsize

target = 'llvm'
ctx = tvm.cpu()

def make_dataset(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def test_resnet(data,batch_size,image_shape):
    num_class=1000
    out_shape = (batch_size, num_class)

    get_time = time.time()
    mod, params = relay.testing.resnet.get_workload(
        num_layers=50, batch_size=batch_size, image_shape=image_shape
    )
    print("-"*10,"Get TVM model",time.time()-get_time,"s","-"*10)

    end_to_end = time.time()
    # build 
    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)
    print("-"*10,"Build_time",(time.time()-end_to_end),"s","-"*10)

    # graph create 
    module = graph_runtime.create(graph, lib, ctx)

    module.run(data = data)
    out = module.get_output(0).asnumpy()

    # lib.export_library(f"./resnet50_{batch_size}_{target}_deploy_lib.tar")
    measurements = 5

    iter_times = []
    print("-"*10,"time.time Module","-"*10)
    for i in range(measurements):
        start_time = time.time()
        module.run(data = data)
        print(f"TVM {model_name}-{batch_size} inference_time : ",(time.time()-start_time)*1000,"ms")
        iter_times.append(time.time() - start_time)

    print(f"TVM model {model_name}-{batch_size} runtime time elapsed",np.mean(iter_times) * 1000 ,"ms")
    print("\n")

    data_tvm = tvm.nd.array(data.astype('float32'))
    e = module.module.time_evaluator("run", ctx, number=measurements, repeat=1)
    t = e(data_tvm).results
    t = np.array(t) * 1000

    print("="*10,"TVM time evaluator module","="*10)
    print('{} (batch={}): {} ms'.format('resnet50', batch_size, t.mean()))

def test_mobilenet(data,batch_size,image_shape):
    mod, params = mobilenet.get_workload( batch_size=batch_size)
    
    end_to_end = time.time()
    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)
    print("build_time",(time.time()-end_to_end)*1000,"ms")

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()
    lib.export_library(f"./mobilenet_{batch_size}_{target}_deploy_lib.tar")
    measurements = 5

    iter_times = []
    print("-"*10,"time.time Module","-"*10)
    for i in range(measurements):
        start_time = time.time()
        module.run(data = data)
        print(f"TVM {model_name}-{batch_size} inference_time : ",(time.time()-start_time)*1000,"ms")
        iter_times.append(time.time() - start_time)

    print(f"TVM model {model_name}-{batch_size} runtime time elapsed",np.mean(iter_times) * 1000 ,"ms")
    print("\n")

    data_tvm = tvm.nd.array(data.astype('float32'))
    e = module.module.time_evaluator("run", ctx, number=5, repeat=1)
    t = e(data_tvm).results
    t = np.array(t) * 1000
    print('{} (batch={}): {} ms'.format('mobilenet', batch_size, t.mean()),"ms")


def test_inception(data,batch_size,image_shape):
    mod, params = inception_v3.get_workload( batch_size=batch_size)
    
    end_to_end = time.time()
    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)
    print("build_time",(time.time()-end_to_end)*1000,"ms")

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()
    # lib.export_library(f"./inception_v3_{batch_size}_{target}_deploy_lib.tar")
    measurements = 5

    iter_times = []
    print("-"*10,"time.time Module","-"*10)
    for i in range(measurements):
        start_time = time.time()
        module.run(data = data)
        print(f"TVM {model_name}-{batch_size} inference_time : ",(time.time()-start_time)*1000,"ms")
        iter_times.append(time.time() - start_time)

    print(f"TVM model {model_name}-{batch_size} runtime time elapsed",np.mean(iter_times) * 1000 ,"ms")
    print("\n")

    data_tvm = tvm.nd.array(data.astype('float32'))
    e = module.module.time_evaluator("run", ctx, number=5, repeat=1)
    t = e(data_tvm).results
    t = np.array(t) * 1000
    print('{} (batch={}): {} ms'.format('inceptionV3', batch_size, t.mean()))

if __name__ == "__main__":
    if model_name =="inception_v3":
        inception_size=299
        inception_data,inception_image_shape=make_dataset(batch_size,inception_size)
    else:
        data,image_shape = make_dataset(batch_size,size)

    if model_name=="resnet50":
        test_resnet(data,batch_size,image_shape)
    if model_name=="mobilenet":
        test_mobilenet(data,batch_size,image_shape)
    if model_name == "inception_v3":
        test_inception(inception_data,batch_size,inception_image_shape)
