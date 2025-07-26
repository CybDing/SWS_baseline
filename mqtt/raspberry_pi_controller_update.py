#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time
import os
import subprocess
import signal
import sys
from datetime import datetime
import urllib.request
import cv2
import numpy as np
import re
import json
import threading
import paho.mqtt.client as mqtt
from PIL import Image
import tflite_runtime.interpreter as tflite
from collections import deque


PORT = '/dev/ttyACM0'
BAUD = 9600
MQTT_BROKER = "192.168.148.250"  # MQTT服务器地址 ipconfig getifaddr en0
MQTT_USERNAME = 'rasp'  
MQTT_PASSWORD = 'sws3009-20-20'  
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "robot/command"
MQTT_TOPIC_DATA = "robot/data"
MQTT_TOPIC_PREDICTION = "robot/prediction"

COMMAND_FREQUENCY = 0.02  # 20ms，50Hz
DATA_FREQUENCY = 0.05     # 50ms，20Hz
SENSOR_READ_FREQUENCY = 0.001  # 10ms，100Hz
IMAGE_CHECK_FREQUENCY = 1.0   # 1s，1Hz - 图像检测频率

SMOOTH_D = 5
SMOOTH_Y = 10

GAIN_D = np.sum([np.exp(-i) for i in range(SMOOTH_D)])
GAIN_Y = np.sum([np.exp(-i) for i in range(SMOOTH_Y)])

photo_dir = './photos'
os.makedirs(photo_dir, exist_ok=True)

CLASS_NAMES = {
    0: "Pallas",
    1: "Persian", 
    2: "Ragdolls",
    3: "Singapura",
    4: "Sphynx"
}

MODEL_PATH = '/home/pi20/Desktop/mqtt_new/CatClassifierV3_14.tflite'  
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

arduino = None
mjpg_process = None
mqtt_client = None
current_command = 'S'
processed_images = set()  # 记录已处理的图像
classifier_interpreter = None
obstacle_detected = False  # 障碍物检测状态

angular_velocity_buffer = deque(maxlen=10)
distance_buffer = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]
angular_velocity = 0.0

# 添加角速度积分相关变量
angular_acceleration_filtered = 0.0
angular_velocity_integrated = 0.0
last_timestamp = None
FILTER_ALPHA = 0.8  # 低通滤波器系数，可调节

data_buffer = {
    'timestamp': '',
    'angular_acceleration': 0,
    'angular_acceleration_filtered': 0,  # 添加滤波后的角加速度
    'angular_velocity': 0,  # 添加积分后的角速度
    'encoder_left': 0,
    'encoder_right': 0,
    'distance_front': 0,
    'distance_left': 0,
    'distance_right': 0,
    'status': 'idle'
}

def load_classifier_model():
    global classifier_interpreter
    
    try:
        if os.path.exists(MODEL_PATH):
            classifier_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            classifier_interpreter.allocate_tensors()
            print(f"[*] 图像分类模型加载成功: {MODEL_PATH}")
            return True
        else:
            print(f"[!] 模型文件不存在: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"[!] 加载分类模型失败: {e}")
        return False

def preprocess_image_for_classification(image_path):
    try:
        input_details = classifier_interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize((input_shape[2], input_shape[1]))
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"[!] 图像预处理失败: {e}")
        return None

def classify_image(image_path):
    global classifier_interpreter
    
    if classifier_interpreter is None:
        return None
    
    try:
        image_array = preprocess_image_for_classification(image_path)
        if image_array is None:
            return None
        
        input_details = classifier_interpreter.get_input_details()
        output_details = classifier_interpreter.get_output_details()
        
        classifier_interpreter.set_tensor(input_details[0]['index'], image_array)
        
        classifier_interpreter.invoke()
        
        output_data = classifier_interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]
        
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        class_name = CLASS_NAMES.get(predicted_class, f"Unknown_{predicted_class}")
        
        return {
            'class_id': int(predicted_class),
            'class_name': class_name,
            'confidence': confidence,
            'all_predictions': predictions.tolist()
        }
        
    except Exception as e:
        print(f"[!] 图像分类失败: {e}")
        return None

def check_new_images():
    global processed_images, mqtt_client
    
    try:
        if not os.path.exists(photo_dir):
            return
        
        current_images = set()
        for file in os.listdir(photo_dir):
            if file.lower().endswith(image_extensions):
                current_images.add(file)
        
        new_images = current_images - processed_images
        
        if new_images:
            print(f"[*] 发现 {len(new_images)} 张新图像")
            
            for image_name in new_images:
                try:
                    image_path = os.path.join(photo_dir, image_name)
                    print(f"[*] Processing image: {image_name}")
                    
                    result = classify_image(image_path)
                    
                    if result:
                        prediction_data = {
                            'timestamp': datetime.now().isoformat(),
                            'filename': image_name,
                            'prediction': result,
                            'status': 'success'
                        }
                        
                        # 发布预测结果到MQTT
                        if mqtt_client and mqtt_client.is_connected():
                            mqtt_client.publish(
                                MQTT_TOPIC_PREDICTION, 
                                json.dumps(prediction_data)
                            )
                        
                        print(f"[*] 预测结果 - 文件: {image_name}")
                        print(f"[*] 类别: {result['class_name']}")
                        print(f"[*] 置信度: {result['confidence']:.4f}")
                        
                    else:
                        # 预测失败
                        error_data = {
                            'timestamp': datetime.now().isoformat(),
                            'filename': image_name,
                            'status': 'error',
                            'message': 'Classification failed'
                        }
                        
                        if mqtt_client and mqtt_client.is_connected():
                            mqtt_client.publish(
                                MQTT_TOPIC_PREDICTION, 
                                json.dumps(error_data)
                            )
                    
                    # 标记为已处理
                    processed_images.add(image_name)
                    
                except Exception as e:
                    print(f"[!] 处理图像 {image_name} 时出错: {e}")
        
        # 清理已删除的图像记录
        processed_images &= current_images
        
    except Exception as e:
        print(f"[!] 检查新图像时出错: {e}")

def start_mjpg_streamer():
    """启动MJPG流媒体服务器"""
    global mjpg_process
    
    try:
        # 先停止可能存在的mjpg_streamer进程
        subprocess.run(["pkill", "-f", "mjpg_streamer"], check=False)
        time.sleep(1)
        
        # 启动新的mjpg_streamer进程
        cmd = [
            "mjpg_streamer",
            "-i", "input_libcamera.so -f 10 -r 320x240 --buffercount 1 -q 30",
            "-o", "output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www"
        ]
        
        mjpg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[*] MJPG Streamer started on port 8080")
        time.sleep(2)  # 等待服务器启动
        
        return True
    except Exception as e:
        print(f"[!] Failed to start MJPG Streamer: {e}")
        return False

def stop_mjpg_streamer():
    """停止MJPG流媒体服务器"""
    global mjpg_process
    
    if mjpg_process:
        mjpg_process.terminate()
        mjpg_process.wait()
        mjpg_process = None
    
    # 确保清理所有mjpg_streamer进程
    subprocess.run(["pkill", "-f", "mjpg_streamer"], check=False)
    print("[*] MJPG Streamer stopped")

def take_photo():
    """通过MJPG Streamer的snapshot接口抓取并保存照片"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(photo_dir, filename)
    
    try:
        url = "http://localhost:8080/?action=snapshot"
        with urllib.request.urlopen(url, timeout=5) as response:
            image_data = response.read()
            image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imwrite(filepath, frame)
                print(f"[*] Photo saved: {filename}")
                return filename
            else:
                print("[!] Failed to decode image")
                return None
    except Exception as e:
        print(f"[!] Snapshot error: {e}")
        return None

def init_serial():
    """初始化串口连接"""
    global arduino
    
    try:
        arduino = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print("[*] Serial connected successfully")
        return True
    except Exception as e:
        print(f"[!] Failed to connect serial: {e}")
        return False

def send_command(command):
    global arduino, current_command
    
    if arduino and arduino.is_open:
        try:
            if command != current_command:
                if obstacle_detected:
                    command = 'S'
                if command == 'S':
                    data_buffer['angular_velocity'] = 0
                arduino.write(command.encode())
                current_command = command
                print(f"[>] Command: {command}")
                data_buffer['status'] = f"Command: {command}"
                return True
        except Exception as e:
            print(f"[!] Sending Command failed: {e}")
            return False
    return False

def read_sensor_data():
    global arduino, data_buffer, obstacle_detected
    
    if arduino and arduino.is_open and arduino.in_waiting:
        try:
            raw = arduino.readline()
            try:
                line = raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                try:
                    line = raw.decode('latin1').strip()
                except:
                    line = raw.decode('utf-8', errors='ignore').strip()
            
            if line.startswith("DIST:"):
                try:
                    matches = re.findall(r'([FLR]):(\d+)', line)
                    values = {k: int(v) for k, v in matches}

                    distance_buffer[0].append(values.get('F', 0))
                    distance_buffer[1].append(values.get('L', 0))
                    distance_buffer[2].append(values.get('R', 0))
                    data_buffer['distance_front'] = np.sum([dis * np.exp(-i) for i, dis in enumerate(distance_buffer[0])]) / GAIN_D
                    data_buffer['distance_left'] = np.sum([dis * np.exp(-i) for i, dis in enumerate(distance_buffer[1])]) / GAIN_D
                    data_buffer['distance_right'] = np.sum([dis * np.exp(-i) for i, dis in enumerate(distance_buffer[2])]) / GAIN_D
                    if data_buffer['distance_front'] < 10:
                        obstacle_detected = True
                        # send_command('S')
                        # data_buffer['status'] = "Auto brake: obstacle detected (< 5cm)"
                        print("Obstacle detected! Auto brake activated!")
                    elif data_buffer['distance_front'] > 10:  
                        obstacle_detected = False
                        # data_buffer['status'] = "Normal operation"
                except Exception as e:
                    pass
            
            # 解析角加速度数据
            elif line.startswith("GYRO:"):
                try:
                    match = re.search(r'GYRO:(-?\d+\.?\d*)', line)
                    if match:
                        current_timestamp = time.time()
                        raw_angular_acceleration = float(match.group(1))
                        
                        # 低通滤波角加速度
                        global angular_acceleration_filtered, angular_velocity_integrated, last_timestamp
                        angular_acceleration_filtered = FILTER_ALPHA * angular_acceleration_filtered + (1 - FILTER_ALPHA) * raw_angular_acceleration
                        
                        # 积分计算角速度
                        if last_timestamp is not None:
                            dt = current_timestamp - last_timestamp
                            if dt > 0 and dt < 1.0:  # 防止异常的时间间隔
                                angular_velocity_integrated += angular_acceleration_filtered * dt
                        
                        last_timestamp = current_timestamp
                        
                        # 更新缓冲区
                        angular_velocity_buffer.append(raw_angular_acceleration)
                        
                        # 更新数据缓冲区
                        data_buffer['angular_acceleration'] = raw_angular_acceleration
                        data_buffer['angular_acceleration_filtered'] = angular_acceleration_filtered
                        data_buffer['angular_velocity'] = angular_velocity_integrated
                        
                except Exception as e:
                    print(f"[!] GYRO数据处理错误: {e}")
            
            elif line.startswith("ENC:"):
                try:
                    matches = re.findall(r'([LR]):(-?\d+)', line)
                    values = {k: int(v) for k, v in matches}
                    data_buffer['encoder_left'] = values.get('L', 0)
                    data_buffer['encoder_right'] = values.get('R', 0)
                    data_buffer['timestamp'] = datetime.now().isoformat()
                    # print(matches)
                except Exception as e:
                    pass
                    
        except Exception as e:
            pass
            # print(f"[!] Error reading sensor data: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT连接回调"""
    if rc == 0:
        print("[*] MQTT connected successfully")
        client.subscribe(MQTT_TOPIC_COMMAND)
        client.subscribe("robot/photo")
    else:
        print(f"[!] MQTT connection failed: {rc}")

def on_mqtt_message(client, userdata, msg):
    global obstacle_detected
    
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic == MQTT_TOPIC_COMMAND:
            # 控制命令
            if payload in ['F', 'B', 'L', 'R', 'S', 'A','B','C','D']:
                # 如果检测到障碍物，强制发送停止命令
                # if obstacle_detected and payload != 'S':
                    print(f"[!] 障碍物检测激活，忽略命令 '{payload}'，强制停止")
                    send_command('S')
                # else:
                    send_command(payload)
            elif payload == 'S':
                send_command('S')
        elif topic == "robot/photo":
            if payload == "take":
                filename = take_photo()
                if filename:
                    client.publish("robot/photo_result", json.dumps({
                        "status": "success",
                        "filename": filename,
                        "timestamp": datetime.now().isoformat()
                    }))
                else:
                    client.publish("robot/photo_result", json.dumps({
                        "status": "error",
                        "message": "Failed to take photo"
                    }))
                    
    except Exception as e:
        print(f"[!] Error processing MQTT message: {e}")

def init_mqtt():
    global mqtt_client
    
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        
        if MQTT_USERNAME and MQTT_PASSWORD:
            mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            print(f"[*] MQTT USERNAME: {MQTT_USERNAME}")
        
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("[*] MQTT client initialized")
        return True
    except Exception as e:
        print(f"[!] Failed to initialize MQTT: {e}")
        return False

def publish_sensor_data():
    global mqtt_client, data_buffer, last_published_data
    
    if mqtt_client:
        try:
            if should_publish_data():
                mqtt_client.publish(MQTT_TOPIC_DATA, json.dumps(data_buffer))
                last_published_data = data_buffer.copy()
        except Exception as e:
            print(f"[!] Failed publishing sensor data: {e}")


last_published_data = {}

def should_publish_data():
    global last_published_data, data_buffer
    
    if not last_published_data:
        return True
    
    threshold = 1  
    for key in ['distance_front', 'distance_left', 'distance_right']:
        if abs(data_buffer.get(key, 0) - last_published_data.get(key, 0)) > threshold:
            return True
    
    if abs(data_buffer.get('encoder_left', 0) - last_published_data.get('encoder_left', 0)) > 5:
        return True
    if abs(data_buffer.get('encoder_right', 0) - last_published_data.get('encoder_right', 0)) > 5:
        return True
    
    return False

def cleanup():
    global arduino, mjpg_process, mqtt_client
    
    print("[*] Cleaning up...")
    
    if arduino and arduino.is_open:
        send_command('S')  
        arduino.close()
        print("[*] Serial connection closed")
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("[*] MQTT client disconnected")
    
    stop_mjpg_streamer()

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n[*] Received interrupt signal")
    cleanup()
    sys.exit(0)

def main():
    print("[*] Starting Raspberry Pi Controller...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not start_mjpg_streamer():
        print("[!] Failed to start MJPG Streamer")
        return
    
    if not init_serial():
        print("[!] Failed to initialize serial connection")
        cleanup()
        return
    
    if not init_mqtt():
        print("[!] Failed to initialize MQTT")
        cleanup()
        return
    
    if not load_classifier_model():
        print("[!] Warning: Image classification model not loaded")
    
    print("[*] All systems initialized successfully")
    print("[*] Robot controller is running...")
    print("[*] Camera stream: http://localhost:8080/stream.html")
    print("[*] Press Ctrl+C to stop")
    
    try:
        last_sensor_read = time.time()
        last_data_publish = time.time()
        last_image_check = time.time()
        
        while True:
            current_time = time.time()
            
            # 高频率读取传感器数据
            if current_time - last_sensor_read >= SENSOR_READ_FREQUENCY:
                read_sensor_data()
                last_sensor_read = current_time
            
            # 中频率发布数据
            if current_time - last_data_publish >= DATA_FREQUENCY:
                publish_sensor_data()
                last_data_publish = current_time
            
            # 低频率检查新图像
            if current_time - last_image_check >= IMAGE_CHECK_FREQUENCY:
                check_new_images()
                last_image_check = current_time
            
            # # 短暂睡眠，避免CPU占用过高
            # time.sleep(0.001)  # 1ms
            
    except KeyboardInterrupt:
        print("\n[*] Interupted by user!")
    except Exception as e:
        print(f"[!] Error occured in: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
