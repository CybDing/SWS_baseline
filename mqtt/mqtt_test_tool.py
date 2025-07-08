#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地MQTT测试工具 - 用于测试MQTT话题和消息
"""

import json
import time
import threading
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = 'main'  # 设置为None则不使用认证，或设置为用户名字符串
MQTT_PASSWORD = 'sws3009-20'  # 设置为None则不使用认证，或设置为密码字符串

# 话题列表
TOPICS = {
    "robot/command": "机器人控制命令",
    "robot/data": "传感器数据",
    "robot/photo": "拍照命令",
    "robot/photo_result": "拍照结果",
    "robot/status": "机器人状态"
}

class MQTTTester:
    def __init__(self):
        self.client = mqtt.Client("mqtt_tester")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.received_messages = {}
        self.is_connected = False
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            print("✅ MQTT连接成功")
            # 订阅所有测试话题
            for topic in TOPICS.keys():
                client.subscribe(topic)
                print(f"📡 订阅话题: {topic}")
        else:
            print(f"❌ MQTT连接失败，错误码: {rc}")
    
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 存储消息
        if topic not in self.received_messages:
            self.received_messages[topic] = []
        
        self.received_messages[topic].append({
            'timestamp': timestamp,
            'payload': payload
        })
        
        # 只保留最近10条消息
        if len(self.received_messages[topic]) > 10:
            self.received_messages[topic].pop(0)
        
        # 打印接收到的消息
        print(f"📨 [{timestamp}] {topic}: {payload}")
    
    def connect(self):
        try:
            # 设置用户名和密码（如果配置了的话）
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
                print(f"🔐 MQTT认证已设置: {MQTT_USERNAME}")
            
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.is_connected = False
        print("🔌 MQTT连接已断开")
    
    def publish_message(self, topic, message):
        if not self.is_connected:
            print("❌ MQTT未连接")
            return False
        
        try:
            self.client.publish(topic, message)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"📤 [{timestamp}] 发送到 {topic}: {message}")
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def get_topic_status(self):
        status = {}
        for topic in TOPICS.keys():
            if topic in self.received_messages:
                last_msg = self.received_messages[topic][-1] if self.received_messages[topic] else None
                status[topic] = {
                    'has_data': len(self.received_messages[topic]) > 0,
                    'message_count': len(self.received_messages[topic]),
                    'last_message': last_msg
                }
            else:
                status[topic] = {
                    'has_data': False,
                    'message_count': 0,
                    'last_message': None
                }
        return status

def print_banner():
    print("="*60)
    print("🧪 MQTT 测试工具")
    print("="*60)
    print("功能:")
    print("  1. 监听所有机器人相关话题")
    print("  2. 发送测试命令")
    print("  3. 查看话题状态")
    print("  4. 实时消息监控")
    print("="*60)

def print_menu():
    print("\n📋 操作菜单:")
    print("1. 发送控制命令")
    print("2. 发送传感器数据")
    print("3. 发送拍照命令")
    print("4. 查看话题状态")
    print("5. 查看最近消息")
    print("6. 清空消息历史")
    print("0. 退出")
    print("-" * 30)

def send_control_command(tester):
    print("\n🎮 控制命令:")
    print("F - 前进, B - 后退, L - 左转, R - 右转, S - 停止")
    command = input("输入命令: ").upper()
    
    if command in ['F', 'B', 'L', 'R', 'S']:
        tester.publish_message("robot/command", command)
    else:
        print("❌ 无效命令")

def send_sensor_data(tester):
    print("\n📊 发送模拟传感器数据")
    import random
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'distance_front': random.randint(20, 100),
        'distance_left': random.randint(15, 80),
        'distance_right': random.randint(15, 80),
        'angular_acceleration': round(random.uniform(-10, 10), 2),
        'encoder_left': random.randint(0, 1000),
        'encoder_right': random.randint(0, 1000),
        'status': 'running'
    }
    
    tester.publish_message("robot/data", json.dumps(data))

def send_photo_command(tester):
    print("\n📸 发送拍照命令")
    tester.publish_message("robot/photo", "take")

def show_topic_status(tester):
    print("\n📊 话题状态:")
    status = tester.get_topic_status()
    
    for topic, info in status.items():
        has_data_icon = "✅" if info['has_data'] else "❌"
        print(f"{has_data_icon} {topic}")
        print(f"   描述: {TOPICS[topic]}")
        print(f"   消息数: {info['message_count']}")
        if info['last_message']:
            print(f"   最后消息: [{info['last_message']['timestamp']}] {info['last_message']['payload'][:50]}...")
        print()

def show_recent_messages(tester):
    print("\n📜 最近消息:")
    for topic, messages in tester.received_messages.items():
        if messages:
            print(f"\n🏷️  {topic}:")
            for msg in messages[-5:]:  # 显示最近5条
                print(f"   [{msg['timestamp']}] {msg['payload']}")

def clear_message_history(tester):
    tester.received_messages.clear()
    print("🗑️  消息历史已清空")

def main():
    print_banner()
    
    # 创建MQTT测试器
    tester = MQTTTester()
    
    # 连接MQTT服务器
    print("🔌 正在连接MQTT服务器...")
    if not tester.connect():
        print("❌ 无法连接到MQTT服务器")
        print("请确保mosquitto服务正在运行:")
        print("   brew install mosquitto")
        print("   brew services start mosquitto")
        return
    
    # 等待连接建立
    time.sleep(2)
    
    try:
        while True:
            print_menu()
            choice = input("请选择操作 (0-6): ").strip()
            
            if choice == '1':
                send_control_command(tester)
            elif choice == '2':
                send_sensor_data(tester)
            elif choice == '3':
                send_photo_command(tester)
            elif choice == '4':
                show_topic_status(tester)
            elif choice == '5':
                show_recent_messages(tester)
            elif choice == '6':
                clear_message_history(tester)
            elif choice == '0':
                break
            else:
                print("❌ 无效选择")
                
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断")
    finally:
        tester.disconnect()
        print("👋 测试结束")

if __name__ == "__main__":
    main()
