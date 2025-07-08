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

# 配置
PORT = '/dev/ttyACM0'
BAUD = 9600
MQTT_BROKER = ""  # MQTT服务器地址 ipconfig getifaddr en0
MQTT_USERNAME = 'rasp'  
MQTT_PASSWORD = 'sws3009-20-20'  
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "robot/command"
MQTT_TOPIC_DATA = "robot/data"

# 频率控制
COMMAND_FREQUENCY = 0.02  # 20ms，50Hz
DATA_FREQUENCY = 0.05     # 50ms，20Hz
SENSOR_READ_FREQUENCY = 0.01  # 10ms，100Hz

# 创建保存照片的目录
photo_dir = './photos'
os.makedirs(photo_dir, exist_ok=True)

# 全局变量
arduino = None
mjpg_process = None
mqtt_client = None
current_command = 'S'

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
            "-i", "input_libcamera.so -f 25 -r 320x240 --buffercount 1 -q 30",
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
                if command != 'S':
                    print(f"[>] 命令: {command}")
                data_buffer['status'] = f"Command: {command}"
                return True
        except Exception as e:
            print(f"[!] 命令发送失败: {e}")
            return False
    return False

def read_sensor_data():
    """读取传感器数据"""
    global arduino, data_buffer
    
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
                    
                    # 自动刹车：前方小于20cm
                    if values.get('F', 1000) < 20:
                        send_command('S')
                        data_buffer['status'] = "Auto brake: obstacle detected"
                except Exception as e:
                    pass
            
            # 解析角加速度数据
            elif line.startswith("GYRO:"):
                try:
                    match = re.search(r'GYRO:(-?\d+\.?\d*)', line)
                    if match:
                        data_buffer['angular_acceleration'] = float(match.group(1))
                except Exception as e:
                    pass
            
            # 解析编码器数据
            elif line.startswith("ENC:"):
                try:
                    matches = re.findall(r'([LR]):(-?\d+)', line)
                    values = {k: int(v) for k, v in matches}
                    data_buffer['encoder_left'] = values.get('L', 0)
                    data_buffer['encoder_right'] = values.get('R', 0)
                    data_buffer['timestamp'] = datetime.now().isoformat()
                except Exception as e:
                    pass
                    
        except Exception as e:
            print(f"[!] Error reading sensor data: {e}")

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
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic == MQTT_TOPIC_COMMAND:
            # 控制命令
            if payload in ['F', 'B', 'L', 'R', 'S']:
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
    
    print("[*] All systems initialized successfully")
    print("[*] Robot controller is running...")
    print("[*] Camera stream: http://localhost:8080/stream.html")
    print("[*] Press Ctrl+C to stop")
    
    # 主循环
    try:
        last_sensor_read = time.time()
        last_data_publish = time.time()
        
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
