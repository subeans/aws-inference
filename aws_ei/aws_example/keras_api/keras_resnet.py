# Resnet Example
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from ei_for_tf.python.keras.ei_keras import EIKerasModel
import numpy as np
import time
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


ITERATIONS = 20

model = ResNet50(weights='imagenet')
ei_model = EIKerasModel(model)
folder_name = os.path.dirname(os.path.abspath(__file__))
img_path = folder_name + '/Serengeti_Elefantenbulle.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# Warm up both models
_ = model.predict(x)
_ = ei_model.predict(x)

# Benchmark both models
for each in range(ITERATIONS):
  start = time.time()
  preds = model.predict(x)
  print("Vanilla iteration %d took %f" % (each, time.time() - start))
for each in range(ITERATIONS):
  start = time.time()
  ei_preds = ei_model.predict(x)
  print("EI iteration %d took %f" % (each, time.time() - start))
  
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
print('EI Predicted:', decode_predictions(ei_preds, top=3)[0])