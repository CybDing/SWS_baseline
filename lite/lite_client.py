import argparse
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from datetime import datetime
import os

CLASS_NAMES = {
    0: "Pallas",
    1: "Persian", 
    2: "Ragdolls",
    3: "Singapura",
    4: "Sphynx"
}

image_base_path = '/path/to/your/images' 
image_extentions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

files = []  

def clear_images(image_base_path):
    global files
    for file in files:
        if file.lower().endswith(image_extentions):
            file_path = os.path.join(image_base_path, file)
            os.remove(file_path)
            print(f"delete {file}")
    files = os.listdir(image_base_path) if os.path.exists(image_base_path) else []

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_cat_breed(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]
    
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return predicted_class, confidence

def detect_new_image():
    global files
    
    if not os.path.exists(image_base_path):
        print(f" {image_base_path} not exists")
        return []
    
    cur_files = os.listdir(image_base_path)
    cur_images = [f for f in cur_files if f.lower().endswith(image_extentions)]
    old_images = [f for f in files if f.lower().endswith(image_extentions)]
    
    new_images = list(set(cur_images) - set(old_images))
    files = cur_files
    
    return new_images

def main():
    global files
    
    parser = argparse.ArgumentParser(description='Cat Classification')
    parser.add_argument('--model', required=True, help='TFLite Model Path')
    
    args = parser.parse_args()
    
    # 初始化文件列表
    if os.path.exists(image_base_path):
        files = os.listdir(image_base_path)
    
    interpreter = load_model(args.model)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    print("开始监控新图片...")
    
    while True:
        new_images = detect_new_image()
        
        if len(new_images) == 0:
            time.sleep(0.3) 
            continue
        
        for image_name in new_images:
            try:
                image_path = os.path.join(image_base_path, image_name)
                print(f"Processing: {image_name}")
                
                image_array = preprocess_image(image_path, input_shape)
                
                predicted_class, confidence = predict_cat_breed(interpreter, image_array)
                
                class_name = CLASS_NAMES.get(predicted_class, f"Unknown_{predicted_class}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"[{timestamp}] File: {image_name}")
                print(f"Category: {class_name}")
                print(f"Confidence: {confidence:.4f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"处理图片 {image_name} 时出错: {e}")
        
        time.sleep(0.1)  

if __name__ == "__main__":
    main()