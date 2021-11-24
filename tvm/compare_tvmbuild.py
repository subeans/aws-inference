import numpy as np

import time 
import tvm
from tvm.contrib import graph_runtime
from tvm import relay
from tvm.relay import testing


def benchmark_execution(mod,
                        params,
                        measure=True,
                        data_shape=(1, 3, 224, 224),
                        out_shape=(1, 1000),
                        dtype='float32'):

    def get_tvm_output(mod, data, params, target, ctx, dtype='float32'):
        with relay.build_config(opt_level=1):
            graph, lib, params = relay.build(mod, target, params=params)

        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", data)
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))

        if measure:
            print("Evaluate graph runtime inference time cost...")
            ftimer = m.module.time_evaluator("run", ctx, number=20, repeat=10)
            # Measure in millisecond.
            prof_res = np.array(ftimer().results) * 1000
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))
            print()
            print("TVM Compiled model - inference time")      
            tvm_result = []
            for i in range(20):
                start = time.time()
                m.run()
                tvm_out = m.get_output(0)
                tvm_result.append(((time.time() - start) * 1000))
                print('tvm compile model :', i, 'th time', (time.time() - start) * 1000,"ms")
            print(np.mean(tvm_result))
        return out.asnumpy()

    def get_tvm_vm_output(mod, data, params, target, ctx, dtype='float32'):
        ex = relay.create_executor('vm', mod=mod, target = target, device=ctx)

        print()
        print("TVM Original model - inference time ")
        results = []
        for i in range(20):
            start_time = time.time()
            result = ex.evaluate()(data, **params)
            results.append(((time.time() - start_time) * 1000))
            print( "original model :",i, 'th time ', ((( time.time() - start_time) ) * 1000),"ms")

        print(np.mean(results))
        # result = ex.evaluate()(data, **params)
        return result.asnumpy().astype(dtype)

    # random input
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = tvm.target.cuda('Tesla_T4')
    ctx = tvm.cuda(0)

    tvm_out = get_tvm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                             target, ctx, dtype)
    vm_out = get_tvm_vm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                               target, ctx, dtype)

    tvm.testing.assert_allclose(vm_out, tvm_out, rtol=1e-5, atol=1e-5)



def test_vgg(batchsize):
    for n in [11, 16]:
        mod, params = tvm.relay.testing.vgg.get_workload(batch_size=batchsize, num_layers=n)
        benchmark_execution(mod, params)


def test_resnet(batchsize):
    for n in [50]:
        mod, params = testing.resnet.get_workload(batch_size=batchsize, num_layers=n)
        benchmark_execution(mod, params, True)



def test_inception_v3(batchsize):
    image_shape = (3, 299, 299)
    mod, params = testing.inception_v3.get_workload(image_shape=image_shape)
    benchmark_execution(mod, params, data_shape=(batchsize, 3, 299, 299))


def test_mobilenet(batchsize):
    mod, params = testing.mobilenet.get_workload(batch_size=batchsize)
    benchmark_execution(mod, params)


def test_densenet(batchsize):
    mod, params = testing.densenet.get_workload(batch_size=batchsize)
    benchmark_execution(mod, params)


if __name__ == '__main__':
    batchsize = 1
    test_resnet(batchsize)
    # test_vgg(batchsize)
    # test_mobilenet(batchsize)
    # test_densenet(batchsize)
    # test_inception_v3(batchsize)
