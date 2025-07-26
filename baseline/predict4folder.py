# # This python code serves the following functions:
# 1. choose a data folders(containing only with unlabeled images) - in main()
# 2. choose a inference model(model_name) - in main()
# 3. show per_image image with predicted category and confidence

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from CatClassifier import CatClassifier
from config import *
from os import listdir
import cv2

def _predict_image(image_path, class_names, show_image=False):
    global model

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return 0.0
    img = cv2.resize(img, IMG_SIZE)
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    if show_image:
        print(image_path)
        plt.figure(figsize=(8, 8))
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(display_img)
        plt.title(f'Predicted Class: {class_names[predicted_class_idx]}\nConfidence: {confidence:.3f}')
        plt.axis('off')
        plt.show()
   
    return confidence

def predict_images(model_name, test_dir):
    global model 
    global class_names

    classifier = CatClassifier(
        data_dir=DATA_DIR,
        img_size=(224, 224),
        batch_size=1
    )
    class_names = sorted(classifier.class_names)
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return None, None
    
    model = classifier.load_model(model_path)
    dir_list = listdir(test_dir)
    index = 0
    AC = 0
    for filename in dir_list:
        image_path = os.path.join(test_dir, filename)
        confidence = _predict_image(
            image_path, 
            class_names,
            show_image=True
        )
        index += 1
        AC = AC + 1 / index * (confidence - AC) 

if __name__ == "__main__":
    test_dir = os.path.join(BASE_DIR, './test_image')
    predict_images(model_name='./model/CatClassifier_512V3_9.h5', test_dir=test_dir)