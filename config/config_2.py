BASE_DIR = '/Users/ding/Desktop/NUS-proj/'
DATA_DIR = '/Users/ding/Desktop/NUS-proj/data/'

IMG_SIZE = (224, 224)
CLASS_NUM = 5
HIDDEN_LAYER_PERCEPTRONS = [512]
DROPOUT = 0.35
CUSTOM_LAYERS = 2 + 3 * len(HIDDEN_LAYER_PERCEPTRONS) + 1
CHECKPOINT_INDEX = 1
FROZEN_LAYERS = 10

# TrainingSet test results:
#               precision    recall  f1-score   support

#       Pallas       0.99      1.00      0.99       760
#      Persian       0.90      0.82      0.85      1366
#     Ragdolls       0.79      0.91      0.84      1098
#    Singapura       0.99      0.94      0.96       666
#       Sphynx       0.99      0.98      0.99      1553

#     accuracy                           0.92      5443
#    macro avg       0.93      0.93      0.93      5443
# weighted avg       0.93      0.92      0.92      5443

# VT results: 0.9014