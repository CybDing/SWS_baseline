# # This python code serves the following functions:
# 1. evaluate(create a confusion matrix) for selected dataset on a selected model

import sys
sys.path.append('./')
from CatClassifier import CatClassifier
from config import *
import os

if __name__ == '__main__':

    data_dir = DATA_DIR
    model_name = 'model/CatClassifierV3_14.h5' 
    model_path = os.path.join(BASE_DIR, model_name) 

    classifier = CatClassifier(
            data_dir= data_dir,
            img_size= IMG_SIZE,
            batch_size=BATCH_SIZE 
    )
    classifier.load_model(model_path)
    print(classifier.model.layers)
    classifier.create_datasets()
    classifier.evaluate(model_name=model_name, evl_dir=os.path.join(DATA_DIR, '../dataset3+2')) # using standard dataset
