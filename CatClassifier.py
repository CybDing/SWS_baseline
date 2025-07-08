# # This python code serves the following functions:
# 1. The base class of classification model
# 2. load_model
# 3. create datasets
# 4. build: build a model (compile...) 
# 5. train: train a model
# 6. add-on features that show loss and acc curve
# *7. evaluate on a standard dataset* (evl_dir should be given) [during training, the test folder from DATA_DIR will be evaluated]

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import model, optim
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import *

def perspective_transform(image):
    """
    使用OpenCV实现透视变换数据增强
    专门为 Keras ImageDataGenerator 设计
    输入: numpy数组 (0-255, uint8)
    输出: numpy数组 (0-255, float32) - 适合后续的 rescale 处理
    """
    # 确保输入是numpy数组
    if hasattr(image, 'shape'):
        img = image.copy()
    else:
        img = np.array(image)
    
    # 确保数据类型为uint8进行处理
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            # 如果是0-1范围，转换为0-255
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    h, w = img.shape[:2]
    
    # 随机决定是否应用透视变换 (50%概率)
    if np.random.random() < 0.5:
        # 定义透视变换的强度
        perspective_strength = 0.1
        
        # 原始四个角点
        src_points = np.float32([
            [0, 0],
            [w, 0], 
            [w, h],
            [0, h]
        ])
        
        # 随机偏移目标点
        max_offset_x = w * perspective_strength
        max_offset_y = h * perspective_strength
        
        dst_points = np.float32([
            [np.random.uniform(-max_offset_x, max_offset_x), 
             np.random.uniform(-max_offset_y, max_offset_y)],
            [w + np.random.uniform(-max_offset_x, max_offset_x), 
             np.random.uniform(-max_offset_y, max_offset_y)],
            [w + np.random.uniform(-max_offset_x, max_offset_x), 
             h + np.random.uniform(-max_offset_y, max_offset_y)],
            [np.random.uniform(-max_offset_x, max_offset_x), 
             h + np.random.uniform(-max_offset_y, max_offset_y)]
        ])
        
        # 计算透视变换矩阵
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            # 应用透视变换
            img = cv2.warpPerspective(img, matrix, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
        except:
            # 如果变换失败，返回原图
            pass
    
    # 转换为float32类型，以便后续的rescale操作
    return img.astype(np.float32)

def advanced_perspective_transform(image):
    """
    更高级的透视变换，包含多种变换类型
    专门为 Keras ImageDataGenerator 设计
    输入: numpy数组 (0-255, uint8)
    输出: numpy数组 (0-255, float32) - 适合后续的 rescale 处理
    """
    if hasattr(image, 'shape'):
        img = image.copy()
    else:
        img = np.array(image)
    
    # 确保数据类型为uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            # 如果是0-1范围，转换为0-255
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    h, w = img.shape[:2]
    
    # 随机选择变换类型
    transform_type = np.random.choice(['perspective', 'trapezoid', 'skew', 'none'], 
                                    p=[0.3, 0.2, 0.2, 0.3])
    
    if transform_type == 'none':
        return img.astype(np.float32)
    
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    if transform_type == 'perspective':
        # 随机透视变换
        offset = min(w, h) * 0.15
        dst_points = np.float32([
            [np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
            [w + np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
            [w + np.random.uniform(-offset, offset), h + np.random.uniform(-offset, offset)],
            [np.random.uniform(-offset, offset), h + np.random.uniform(-offset, offset)]
        ])
    
    elif transform_type == 'trapezoid':
        # 梯形变换
        offset = min(w, h) * 0.1
        dst_points = np.float32([
            [offset, 0],
            [w - offset, 0],
            [w, h],
            [0, h]
        ])
    
    elif transform_type == 'skew':
        # 倾斜变换
        skew_x = np.random.uniform(-0.2, 0.2) * w
        skew_y = np.random.uniform(-0.2, 0.2) * h
        dst_points = np.float32([
            [skew_x, skew_y],
            [w + skew_x, skew_y],
            [w, h],
            [0, h]
        ])
    
    try:
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img = cv2.warpPerspective(img, matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
    except:
        pass
    
    return img.astype(np.float32)

def test_perspective_transform(image_path, save_path=None):
    """
    测试透视变换效果，生成对比图
    
    Args:
        image_path: 测试图片路径
        save_path: 保存路径（可选）
    """
    import cv2
    import matplotlib.pyplot as plt
    
    # 读取图片
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 应用不同的透视变换
    simple_transform = perspective_transform(img_rgb)
    advanced_transform = advanced_perspective_transform(img_rgb)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(simple_transform/255.0)
    axes[1].set_title('Simple Perspective Transform')
    axes[1].axis('off')
    
    axes[2].imshow(advanced_transform/255.0)
    axes[2].set_title('Advanced Perspective Transform')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

class CatClassifier:

    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, perspective_type='advanced'):
        """
        初始化猫分类器
        
        Args:
            data_dir: 数据目录
            img_size: 图像尺寸
            batch_size: 批量大小
            perspective_type: 透视变换类型 ('simple', 'advanced', 'none')
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.perspective_type = perspective_type
        self.model = None
        self.history = None

        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            self.class_names = [d for d in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, d)) ]
        else:
            raise ValueError(f"data_dir not found: {train_dir}")
        
        print(f"List of categories: {self.class_names}")
        print(f"Total {len(self.class_names)} categories")

        # 选择透视变换函数
        if perspective_type == 'simple':
            preprocessing_func = perspective_transform
        elif perspective_type == 'advanced':
            preprocessing_func = advanced_perspective_transform
        else:
            preprocessing_func = None

        # 训练集
        self.train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='constant',
            width_shift_range=0.1,
            height_shift_range=0.1,
            cval=255,
            preprocessing_function=preprocessing_func,  # 使用自定义透视变换
            rescale=1/255.0,  # 将 rescale 放在最后
        )

        # 验证集（一般不做增强，只归一化）
        self.val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

    def create_datasets(self):

        # def custom_preprocessing(img):
        #     scale_b = np.random.uniform(0.95, 1.2)
        #     scale_g = np.random.uniform(0.99, 1.1)
        #     scale_r = 1
        #     img[:, :, 2] = np.clip(img[:, :, 2] * scale_b , 0, 255) 
        #     img[:, :, 0] = np.clip(img[:, :, 0] * scale_r , 0, 255)  
        #     img[:, :, 1] = np.clip(img[:, :, 1] * scale_g , 0, 255) 
        #     return img

        # def simple_white_balance(img):
        #     import cv2
        #     """使用OpenCV的简单白平衡"""
        #     if img.max() <= 1.0:
        #         img = img * 255.0
        #     img = img.astype(np.uint8)
            
        #     wb = cv2.xphoto.createSimpleWB()
        #     balanced_img = wb.balanceWhite(img)
            
        #     return balanced_img.astype(np.float32) / 255.0

        # def custom_preprocessing(img):
        #     # 随机决定是否应用白平衡（30%概率）
        #     if np.random.random() < 0.3:
        #         img = simple_white_balance(img)
        #     else:
        #         # 标准化到0-1范围
        #         if img.max() > 1.0:
        #             img = img / 255.0
            
        #     return img
    
        train_data_dir = os.path.join(self.data_dir, 'train')
        val_data_dir = os.path.join(self.data_dir, 'validation')
        
        if not os.path.exists(train_data_dir):
            raise ValueError(f"训练数据目录不存在: {train_data_dir}")
        if not os.path.exists(val_data_dir):
            raise ValueError(f"验证数据目录不存在: {val_data_dir}")
        
        self.train_dataset = self.train_gen.flow_from_directory(
            train_data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        print(f"Training data: total {self.train_dataset.samples} images, {len(self.train_dataset.class_indices)} categories")
        
        self.val_dataset = self.val_gen.flow_from_directory(
            val_data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False  
        )
        
        print(f"Testing data: total {self.val_dataset.samples} images, {len(self.val_dataset.class_indices)} categories")
        return self.train_dataset, self.val_dataset


    def build(self):
        self.model = model

        self.model.compile(
            optimizer=optim,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
        return self.model
    
    def train(self, save_path=None, continued=False):

        if self.model is None:
            raise ValueError("No model available!")
        
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("No generator available!")
        
        assert save_path is not None
        save_path = os.path.join(BASE_DIR, save_path)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=4,
                min_lr=1e-8
            ),
            tf.keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        print("***** 1 Training Dense Layer only *****")
        for layer in self.model.layers[:-CUSTOM_LAYERS]:
            layer.trainable = False

        self.history = self.model.fit(
            self.train_dataset,
            epochs=int(EPOCHS/continued),
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        print(f"***** 2 Training pretrained models layers(from {FROZEN_LAYERS}) with Hidden layers({HIDDEN_LAYER_PERCEPTRONS}) *****")
        for layer in self.model.layers[-FROZEN_LAYERS-CUSTOM_LAYERS:]:
            layer.trainable = True

        fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  
            patience=3,
            min_lr=1e-8
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
        ]

        self.history = self.model.fit(
            self.train_dataset,
            epochs=int((EPOCHS/1.5)/continued),
            validation_data=self.val_dataset,
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        print(f"Training Complete! Model Saved to: {save_path}")
        return self.history
    
    def plot_training_history(self):
        
        if self.history is None:
            print("No training history found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, evl_dir=None, model_name = None):
        if self.model is None:
            print("Please create model first!")
            return
        
        path_cut = None

        if evl_dir is None:
            evl_dir = os.path.join(BASE_DIR, 'validation')
            path_cut = 'validation'

        elif os.path.exists(evl_dir) == False:
            raise ValueError(f"evl_dir {evl_dir} not found!")
        
        else:
            path_cut = evl_dir
        
        evl_dataset = self.val_gen.flow_from_directory(
            evl_dir,
            target_size = self.img_size,
            batch_size = self.batch_size,
            class_mode = 'categorical',
            shuffle = False,
        )

        print(f"Testing data: total {evl_dataset.samples} images, {len(evl_dataset.class_indices)} categories")
        
        print(f"Evaluating {path_cut}")  
        
        predictions = self.model.predict(evl_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = evl_dataset.classes
        class_labels = list(evl_dataset.class_indices.keys())
        
        print("\nTrainingSet test results:")  
        print(classification_report(true_classes, predicted_classes, 
                                target_names=class_labels))
        
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
        string = 'Confusion Matrix' if model_name is None \
            else f'Confusion Matrix - {model_name}'
        plt.title(string) 
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"\n{path_cut} Accuaracy: {accuracy:.4f}")  #
        
        return predictions, true_classes
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.class_mapping = sorted(self.class_names)
        
        print(f"Model is loaded from {model_path} and compiled")
        print(f"Mapping: {dict(enumerate(self.class_mapping))}")
        
        return self.model
