BASE_DIR = '/Users/ding/Desktop/NUS-proj/'
DATA_DIR = '/Users/ding/Desktop/NUS-proj/data/mdata_new'

# IMG_SIZE = (240, 320) # H * W
IMG_SIZE = (256, 256) # H * W
CLASS_NUM = 5
HIDDEN_LAYER_PERCEPTRONS = [512]
DROPOUT = 0.4
CUSTOM_LAYERS = 2 + 3 * len(HIDDEN_LAYER_PERCEPTRONS) + 1
CHECKPOINT_INDEX = 20
FROZEN_LAYERS = 10
BATCH_SIZE = 32
# MODEL = 'NASNETMOBILE'
MODEL = 'INCEPTIONV3'
EPOCHS = 30
# Dataset Balanced+
# Use white color for padding...