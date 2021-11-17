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

# resnet50
model = 'resnet50'
mod, params = relay.testing.resnet.get_workload(
    num_layers=50, batch_size=batch_size, image_shape=image_shape
)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

# compilation
opt_level = 3
target = "llvm"

with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)


# run general 
# create random input
dev = tvm.cpu(0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
data_tvm = tvm.nd.array(data.astype('float32'))


# create module
mm = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
mm.set_input("data", data_tvm)
# run
mm.run()
# get output
out = mm.get_output(0, tvm.nd.empty(out_shape)).numpy()

# time_evaluator 
# time_evaluator ( func_name , dev , number = 10 , repeat = 1 , min_repeat_ms = 0 , f_preproc = '' )
# 기능 실행의 시간 비용을 측정하는 평가자를 가져옵니다.
# 매개변수
# func_name ( str ) – 모듈에 있는 함수의 이름입니다.
# dev ( Device ) – 이 기능을 실행해야 하는 장치입니다.
# number ( int ) – 평균을 구하기 위해 이 함수를 실행할 횟수입니다. 이러한 실행 을 측정의 1회 반복 이라고 합니다.
# repeat ( int , optional ) – 측정을 반복할 횟수입니다. 전체적으로 함수는 (1 + 숫자 x 반복) 호출되며 첫 번째 함수는 워밍업되어 폐기됩니다. 반환된 결과에는 반복 비용이 포함 되며 각 비용 은 수 비용 의 평균입니다 .
# min_repeat_ms ( int , optional ) – 1회 반복 의 최소 ​​지속 시간( 밀리초)입니다. 기본적으로 하나의 반복 에는 숫자 실행이 포함 됩니다. 이 매개변수가 설정되면 매개변수 번호 는 1회 반복 의 최소 ​​지속 시간 요구 사항을 충족하도록 동적으로 조정됩니다 . 즉, 한 번의 반복 실행 시간 이 이 시간 이하로 떨어지면 숫자 매개변수가 자동으로 증가합니다.
# f_preproc ( str , optional ) – 시간 평가기를 실행하기 전에 실행하려는 전처리 함수 이름입니다.

e = mm.module.time_evaluator("run", dev, number=10, repeat=3)
t = e(data_tvm).results
t = np.array(t) * 1000

print('{} (batch={}): {} ms'.format(model, batch_size, t.mean()))

# Print first 10 elements of output
print(out.flatten())
print(len(out.flatten()))

# saved compiled module
from tvm.contrib import utils

lib.export_library(f"./{model}_deploy_lib.tar")
