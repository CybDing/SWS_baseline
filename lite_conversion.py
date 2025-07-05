# # This python code serves the following functions:
# 1. transform your .h5 model into .tflite (TESTED only on MAC only!)

import tensorflow as tf
import os
from config import *

model_name = './model/CatClassifier_512V3_9.h5'
model_path = os.path.join(BASE_DIR, model_name)
saved_model_dir = os.path.join(BASE_DIR, 'saved_model')

model = tf.keras.models.load_model(model_path)

model.export(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('./model_lite/CatClassifier_V3_9.tflite', 'wb') as f:
    f.write(tflite_model)

