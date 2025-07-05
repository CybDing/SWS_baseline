BASE_DIR = '/Users/ding/Desktop/NUS-proj/'
DATA_DIR = '/Users/ding/Desktop/NUS-proj/data/'

IMG_SIZE = (224, 224)
CLASS_NUM = 5
HIDDEN_LAYER_PERCEPTRONS = [512]
DROPOUT = 0.3
CUSTOM_LAYERS = 2 + 3 * len(HIDDEN_LAYER_PERCEPTRONS) + 1
CHECKPOINT_INDEX = 1

# Good at Persion, not good at ragdoll 
