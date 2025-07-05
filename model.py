import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import *

if MODEL == 'NASNETMOBILE':
    base_model = tf.keras.applications.NASNetMobile(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
elif MODEL == 'MOBILENETV2':
    base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
else: raise ValueError("MODEL NAME PARSING ERROR!")

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT)(x)

for neurons in HIDDEN_LAYER_PERCEPTRONS:
    x = layers.Dense(neurons, activation='relu')(x) #  
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

predictions = layers.Dense(CLASS_NUM, activation='softmax')(x)

model = models.Model(inputs = base_model.input, outputs = predictions)

optim = RMSprop()