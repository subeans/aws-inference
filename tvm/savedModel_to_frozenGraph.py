import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
args = parser.parse_args()
model_name = args.model

saved_input_dir = f"./{model_name}_saved_model"

model = tf.saved_model.load(saved_input_dir)
graph_func = model.signatures['serving_default']
frozen_func = convert_variables_to_constants_v2(graph_func)
frozen_func.graph.as_graph_def()


# saved_model layer정보 확인 
# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 50)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)
# print("-" * 50)

print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name=f"frozen_{model_name}.pb",
                      as_text=False)



