# This python code serves the following functions:
# 1. Peek at the augmented images from train_gen;
# 2. print out 9 images from a batch to preview.

from CatClassifier import CatClassifier
import matplotlib.pyplot as plt
import numpy as np
from config import *

def preview_augmented_images():
    data_dir = DATA_DIR
    classifier = CatClassifier(
        data_dir=data_dir,
        img_size = IMG_SIZE,
        batch_size=9  #
    )
    train_gen, _ = classifier.create_datasets()
    images, _ = next(train_gen) 
    plt.figure(figsize=(8, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.suptitle("Images with data-augmentation")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_augmented_images()