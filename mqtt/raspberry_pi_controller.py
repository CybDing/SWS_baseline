#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派控制器 - 负责控制小车和传感器数据收集
"""

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
# 添加图像分类相关导入
from PIL import Image
import tflite_runtime.interpreter as tflite

# 配置
PORT = '/dev/ttyACM0'
BAUD = 9600
MQTT_BROKER = "192.168.148.250"  # MQTT服务器地址 ipconfig getifaddr en0
MQTT_USERNAME = 'rasp'  
MQTT_PASSWORD = 'sws3009-20-20'  
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "robot/command"
MQTT_TOPIC_DATA = "robot/data"
# 添加新的MQTT主题
MQTT_TOPIC_PREDICTION = "robot/prediction"

# 频率控制
COMMAND_FREQUENCY = 0.02  # 20ms，50Hz
DATA_FREQUENCY = 0.05     # 50ms，20Hz
SENSOR_READ_FREQUENCY = 0.01  # 10ms，100Hz
IMAGE_CHECK_FREQUENCY = 1.0   # 1s，1Hz - 图像检测频率

# 创建保存照片的目录
photo_dir = './photos'
os.makedirs(photo_dir, exist_ok=True)

# 图像分类配置
CLASS_NAMES = {
    0: "Pallas",
    1: "Persian", 
    2: "Ragdolls",
    3: "Singapura",
    4: "Sphynx"
}

MODEL_PATH = '/home/pi20/Desktop/mqtt_new/CatClassifierV3_14.tflite'  
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# 全局变量
arduino = None
mjpg_process = None
mqtt_client = None
current_command = 'S'
processed_images = set()  # 记录已处理的图像
classifier_interpreter = None
obstacle_detected = False  # 障碍物检测状态

# 数据缓冲区
data_buffer = {
    'timestamp': '',
    'angular_acceleration': 0,
    'encoder_left': 0,
    'encoder_right': 0,
    'distance_front': 0,
    'distance_left': 0,
    'distance_right': 0,
    'status': 'idle'
}

def load_classifier_model():
    """加载图像分类模型"""
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
    """预处理图像用于分类"""
    try:
        # 获取模型输入尺寸
        input_details = classifier_interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        # 加载和预处理图像
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
    """对图像进行分类预测"""
    global classifier_interpreter
    
    if classifier_interpreter is None:
        return None
    
    try:
        # 预处理图像
        image_array = preprocess_image_for_classification(image_path)
        if image_array is None:
            return None
        
        # 获取输入输出详情
        input_details = classifier_interpreter.get_input_details()
        output_details = classifier_interpreter.get_output_details()
        
        # 设置输入数据
        classifier_interpreter.set_tensor(input_details[0]['index'], image_array)
        
        # 执行推理
        classifier_interpreter.invoke()
        
        # 获取预测结果
        output_data = classifier_interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]
        
        # 获取预测类别和置信度
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
    """检查新图像并进行分类"""
    global processed_images, mqtt_client
    
    try:
        if not os.path.exists(photo_dir):
            return
        
        # 获取目录中的所有图像文件
        current_images = set()
        for file in os.listdir(photo_dir):
            if file.lower().endswith(image_extensions):
                current_images.add(file)
        
        # 找出新图像
        new_images = current_images - processed_images
        
        if new_images:
            print(f"[*] 发现 {len(new_images)} 张新图像")
            
            for image_name in new_images:
                try:
                    image_path = os.path.join(photo_dir, image_name)
                    print(f"[*] 正在处理图像: {image_name}")
                    
                    # 进行分类预测
                    result = classify_image(image_path)
                    
                    if result:
                        # 构建预测结果消息
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
    """发送命令到Arduino"""
    global arduino, current_command
    
    if arduino and arduino.is_open:
        try:
            if command != current_command:
                arduino.write(command.encode())
                current_command = command
                # 只记录非停止命令
                # if command != 'S':
                print(f"[>] 命令: {command}")
                data_buffer['status'] = f"Command: {command}"
                return True
        except Exception as e:
            print(f"[!] 命令发送失败: {e}")
            return False
    return False

def read_sensor_data():
    """读取传感器数据"""
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
            
            # 解析距离传感器数据
            if line.startswith("DIST:"):
                try:
                    matches = re.findall(r'([FLR]):(\d+)', line)
                    values = {k: int(v) for k, v in matches}
                    data_buffer['distance_front'] = values.get('F', 0)
                    data_buffer['distance_left'] = values.get('L', 0)
                    data_buffer['distance_right'] = values.get('R', 0)
                    
                    # 检查前方障碍物
                    front_distance = values.get('F', 1000)
                    if front_distance < 5:
                        obstacle_detected = True
                        send_command('S')
                        data_buffer['status'] = "Auto brake: obstacle detected (< 5cm)"
                        print("Obstacle detected! Auto brake activated!")
                    elif front_distance > 10:  # 增加一些回滞，避免频繁切换
                        obstacle_detected = False
                        data_buffer['status'] = "Normal operation"
                        
                except Exception as e:
                    pass
            
            # 解析角加速度数据
            elif line.startswith("GYRO:"):
                try:
                    match = re.search(r'GYRO:(-?\d+\.?\d*)', line)
                    if match:
                        data_buffer['angular_acceleration'] = float(match.group(1))
                except Exception as e:
                    print(e)            
      
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
    """MQTT消息回调"""
    global obstacle_detected
    
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic == MQTT_TOPIC_COMMAND:
            # 控制命令
            if payload in ['F', 'B', 'L', 'R', 'S']:
                # 如果检测到障碍物，强制发送停止命令
                if obstacle_detected and payload != 'S':
                    print(f"[!] 障碍物检测激活，忽略命令 '{payload}'，强制停止")
                    send_command('S')
                else:
                    send_command(payload)
            elif payload == 'STOP':
                send_command('S')
        elif topic == "robot/photo":
            # 拍照命令
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
    """初始化MQTT客户端"""
    global mqtt_client
    
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        
        # 设置用户名和密码（如果配置了的话）
        if MQTT_USERNAME and MQTT_PASSWORD:
            mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            print(f"[*] MQTT认证已设置: {MQTT_USERNAME}")
        
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("[*] MQTT client initialized")
        return True
    except Exception as e:
        print(f"[!] Failed to initialize MQTT: {e}")
        return False

def publish_sensor_data():
    """发布传感器数据到MQTT"""
    global mqtt_client, data_buffer, last_published_data
    
    if mqtt_client:
        try:
            # 只在数据有变化时发布，减少网络负载
            if should_publish_data():
                mqtt_client.publish(MQTT_TOPIC_DATA, json.dumps(data_buffer))
                last_published_data = data_buffer.copy()
        except Exception as e:
            print(f"[!] 传感器数据发布失败: {e}")

# 上次发布的数据
last_published_data = {}

def should_publish_data():
    """判断是否应该发布数据"""
    global last_published_data, data_buffer
    
    # 如果是第一次发布，直接发布
    if not last_published_data:
        return True
    
    # 检查关键数据是否有变化
    threshold = 2  # 2cm的变化阈值
    for key in ['distance_front', 'distance_left', 'distance_right']:
        if abs(data_buffer.get(key, 0) - last_published_data.get(key, 0)) > threshold:
            return True
    
    # 检查编码器数据是否有显著变化
    if abs(data_buffer.get('encoder_left', 0) - last_published_data.get('encoder_left', 0)) > 5:
        return True
    if abs(data_buffer.get('encoder_right', 0) - last_published_data.get('encoder_right', 0)) > 5:
        return True
    
    return False

def cleanup():
    """清理资源"""
    global arduino, mjpg_process, mqtt_client
    
    print("[*] Cleaning up...")
    
    if arduino and arduino.is_open:
        send_command('S')  # 停止小车
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
    """主函数"""
    print("[*] Starting Raspberry Pi Controller...")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化组件
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
    
    # 加载图像分类模型
    if not load_classifier_model():
        print("[!] Warning: Image classification model not loaded")
    
    print("[*] All systems initialized successfully")
    print("[*] Robot controller is running...")
    print("[*] Camera stream: http://localhost:8080/stream.html")
    print("[*] Press Ctrl+C to stop")
    
    # 主循环
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
            
            # 短暂睡眠，避免CPU占用过高
            time.sleep(0.001)  # 1ms
            
    except KeyboardInterrupt:
        print("\n[*] 用户中断")
    except Exception as e:
        print(f"[!] 意外错误: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
