#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MQTT监听器 - 专门用于监听Web界面发送的命令
"""

import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# MQTT配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = 'main'
MQTT_PASSWORD = 'sws3009-20'

# 话题列表
TOPICS = [
    "robot/command",
    "robot/data",
    "robot/photo",
    "robot/photo_result",
    "robot/status"
]

class MQTTListener:
    def __init__(self):
        self.client = mqtt.Client("web_mqtt_listener")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.message_count = 0
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT连接成功")
            print("🔐 MQTT认证已设置: main")
            print("📡 开始监听以下话题:")
            
            # 订阅所有话题
            for topic in TOPICS:
                client.subscribe(topic)
                print(f"   • {topic}")
            
            print("\n🎯 等待Web界面发送命令...")
            print("💡 请在Web界面按 W/A/S/D 键或点击按钮")
            print("=" * 50)
        else:
            print(f"❌ MQTT连接失败，错误码: {rc}")
    
    def on_message(self, client, userdata, msg):
        self.message_count += 1
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print(f"\n📨 [{timestamp}] 收到消息 #{self.message_count}")
        print(f"📋 话题: {topic}")
        print(f"📄 内容: {payload}")
        
        # 解析控制命令
        if topic == "robot/command":
            command_names = {
                'F': '前进 ↑',
                'B': '后退 ↓', 
                'L': '左转 ←',
                'R': '右转 →',
                'S': '停止 ⏹️'
            }
            command_name = command_names.get(payload, payload)
            print(f"🎮 控制命令: {command_name}")
            
        elif topic == "robot/photo":
            print(f"📸 拍照命令: {payload}")
            
        print("-" * 50)
    
    def connect(self):
        try:
            # 设置用户名和密码
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def run(self):
        if not self.connect():
            return
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 停止监听")
            self.client.loop_stop()
            self.client.disconnect()
            print(f"📊 共接收到 {self.message_count} 条消息")

def main():
    print("=" * 60)
    print("🎯 Web界面MQTT命令监听器")
    print("=" * 60)
    
    listener = MQTTListener()
    listener.run()

if __name__ == "__main__":
    main()
