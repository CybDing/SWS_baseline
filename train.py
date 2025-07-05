# # This python code serves the following functions:
# 1. *** write all of your parameters of the model, other info, into config.py, all changes will automatically parsed
# 2. A config.py will save into ./config/config_x, which x is the _x.h5 index
# 3. Train a model, and save to model_name under BASE_DIR.


from CatClassifier import CatClassifier
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import *

import shutil

def main(model_name):
    import config
    checkpoint_index = config.CHECKPOINT_INDEX + 1

    config_backup_path = f'./config/config_{checkpoint_index}.py'
    shutil.copy('config.py', config_backup_path)
    print(f"Config backed up to {config_backup_path}")

    with open('./config.py', 'r') as f:
        lines = f.readlines()
    with open('./config.py', 'w') as f:
        for line in lines:
            if line.startswith('CHECKPOINT_INDEX'):
                f.write(f'CHECKPOINT_INDEX = {checkpoint_index}\n')
            else:
                f.write(line)

    data_dir = DATA_DIR
    classifier = CatClassifier(
        data_dir=data_dir,
        img_size=IMG_SIZE,
        batch_size=32  
    )
    classifier.train_dataset, classifier.val_dataset = classifier.create_datasets()
    print(f"Loading existing model: {model_name}")

    save_path = f'{model_name}_{checkpoint_index}.h5'
    print(f"Training will save to: {save_path}")

    continued = 1
    try:
        classifier.model = classifier.load_model(save_path)
        print("Model is loaded successfully!")
        continued = 2
    except Exception as e:
        flag = input(f"No model found in {save_path}. Create new model instead? y/n")
        if flag == 'y': 
            print("New model created!")
        else: 
            print("Program stopped!")
            return
    classifier.model = classifier.build()

    _ = classifier.train(save_path=save_path, continued=continued)
    classifier.plot_training_history()
    classifier.evaluate()

if __name__ == "__main__":
    main(model_name='./Model/CatClassifierV3')