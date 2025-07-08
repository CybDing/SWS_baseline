import serial
import time
import curses
import os
from datetime import datetime
import urllib.request
import cv2
import numpy as np
import re
import json
import threading
import http.server
import socketserver

# import IMU.fusing_new
# from IMU.fusing_new import IMU

# imu = IMU()
# imu.restart()
PORT = '/dev/ttyACM0'
BAUD = 9600

# 创建保存照片的目录
photo_dir = './photos'
os.makedirs(photo_dir, exist_ok=True)

# 数据传输相关
data_buffer = {
    'timestamp': '',
    'angular_acceleration': 0,
    'encoder_left': 0,
    'encoder_right': 0,
    'distance_front': 0,
    'distance_left': 0,
    'distance_right': 0
}

# 超声波稳定性检测
class UltrasonicStabilityDetector:
    def __init__(self, window_size=8, threshold=3):  # 减小窗口大小，减小阈值
        self.window_size = window_size
        self.threshold = threshold
        self.front_readings = []
        self.left_readings = []
        self.right_readings = []
    
    def add_reading(self, front, left, right):
        """添加新的距离读数"""
        self.front_readings.append(front)
        self.left_readings.append(left)
        self.right_readings.append(right)
        
        # 保持窗口大小
        if len(self.front_readings) > self.window_size:
            self.front_readings.pop(0)
        if len(self.left_readings) > self.window_size:
            self.left_readings.pop(0)
        if len(self.right_readings) > self.window_size:
            self.right_readings.pop(0)
    
    def is_stable(self, readings):
        """检查读数是否稳定"""
        if len(readings) < 5:  # 至少需要5个读数
            return False
        
        # 计算最近读数的平均值和标准差
        recent_readings = readings[-5:]  # 取最近5个读数
        avg = sum(recent_readings) / len(recent_readings)
        
        # 计算标准差
        variance = sum((x - avg) ** 2 for x in recent_readings) / len(recent_readings)
        std_dev = variance ** 0.5
        
        # 检查标准差是否在阈值内（更严格的稳定性判断）
        if std_dev > self.threshold:
            return False
        
        # 额外检查：最新的3个读数都必须在平均值±阈值范围内
        for reading in recent_readings[-3:]:
            if abs(reading - avg) > self.threshold:
                return False
        
        return True
    
    def get_stable_distances(self):
        """获取稳定的距离值"""
        front_stable = self.is_stable(self.front_readings)
        left_stable = self.is_stable(self.left_readings)
        right_stable = self.is_stable(self.right_readings)
        
        return {
            'front': self.front_readings[-1] if front_stable and self.front_readings else None,
            'left': self.left_readings[-1] if left_stable and self.left_readings else None,
            'right': self.right_readings[-1] if right_stable and self.right_readings else None,
            'front_stable': front_stable,
            'left_stable': left_stable,
            'right_stable': right_stable
        }

# 创建稳定性检测器（更严格的参数）
stability_detector = UltrasonicStabilityDetector(window_size=8, threshold=2)

class DataHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data_buffer).encode())
        else:
            super().do_GET()

def start_http_server():
    """启动HTTP服务器在后台线程"""
    port = 8081
    handler = DataHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"[*] HTTP server started on port {port}")
    httpd.serve_forever()

# 启动HTTP服务器线程
server_thread = threading.Thread(target=start_http_server, daemon=True)
server_thread.start()

# 打开串口
try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("[*] Serial connected. Use arrow keys to control. Press X to take photo. Press ESC to exit.")
except Exception as e:
    print(f"[!] Failed to connect: {e}")
    exit()


def take_photo():
    """通过 MJPG Streamer 的 snapshot 接口抓取并保存照片"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(photo_dir, filename)

    try:
        url = "http://localhost:8080/?action=snapshot"
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
            image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imwrite(filepath, frame)
                return filepath
            else:
                return None
    except Exception as e:
        print(f"[!] Snapshot error: {e}")
        return None


def main(stdscr):
    global data_buffer
    
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.curs_set(0)

    last_command = ''
    current_command = 'S'
    direction_locked = False

    distF = "N/A"
    distL = "N/A"
    distR = "N/A"

    angular_acceleration = 0
    encoder_left = 0
    encoder_right = 0

    stdscr.addstr(0, 0, "[*] Use arrow keys to control. Press X to take photo. Press ESC to exit.")
    stdscr.refresh()

    while True:
        try:
            key = stdscr.getch()

            if key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
                if not direction_locked:
                    if key == curses.KEY_UP:
                        current_command = 'F'
                    elif key == curses.KEY_DOWN:
                        current_command = 'B'
                    elif key == curses.KEY_LEFT:
                        current_command = 'L'
                    elif key == curses.KEY_RIGHT:
                        current_command = 'R'
                    direction_locked = True

            elif key == -1:
                if direction_locked:
                    current_command = 'S'
                    direction_locked = False

            elif key in [ord('x'), ord('X')]:
                filepath = take_photo()
                stdscr.move(2, 0)
                stdscr.clrtoeol()
                if filepath:
                    stdscr.addstr(2, 0, f"[*] Photo saved: {filepath}")
                else:
                    stdscr.addstr(2, 0, "[!] Failed to capture photo.")
                stdscr.refresh()

            elif key == 27:
                stdscr.addstr(4, 0, "[*] Exit requested. Closing serial...")
                stdscr.refresh()
                break

            # 读取串口中的传感器信息
            if arduino.in_waiting:
                raw = arduino.readline()
                try:
                    line = raw.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # 尝试使用 latin1 或忽略非法字符
                    try:
                        line = raw.decode('latin1').strip()
                    except:
                        line = raw.decode('utf-8', errors='ignore').strip()
                
                # 解析距离传感器数据
                if line.startswith("DIST:"):
                    try:
                        # 正则提取三个方向的距离
                        matches = re.findall(r'([FLR]):(\d+)', line)
                        values = {k: int(v) for k, v in matches}
                        
                        # 添加读数到稳定性检测器
                        stability_detector.add_reading(
                            values.get('F', 0),
                            values.get('L', 0),
                            values.get('R', 0)
                        )
                        
                        # 获取稳定的距离值
                        stable_distances = stability_detector.get_stable_distances()
                        
                        # 更新显示值
                        distF = f"{stable_distances['front']} cm" if stable_distances['front'] is not None else "N/A"
                        distL = f"{stable_distances['left']} cm" if stable_distances['left'] is not None else "N/A"
                        distR = f"{stable_distances['right']} cm" if stable_distances['right'] is not None else "N/A"

                        # 更新数据缓冲区（使用稳定值或0）
                        data_buffer['distance_front'] = stable_distances['front'] if stable_distances['front'] is not None else 0
                        data_buffer['distance_left'] = stable_distances['left'] if stable_distances['left'] is not None else 0
                        data_buffer['distance_right'] = stable_distances['right'] if stable_distances['right'] is not None else 0

                        # 自动刹车：前方小于 20cm（只在稳定时生效）
                        if stable_distances['front'] is not None and stable_distances['front'] < 20:
                            current_command = 'S'
                            direction_locked = False
                    except Exception as e:
                        pass  # 忽略解析失败
                
                # 解析角加速度数据
                elif line.startswith("GYRO:"):
                    try:
                        # 格式: GYRO:123.45
                        match = re.search(r'GYRO:(-?\d+\.?\d*)', line)
                        if match:
                            angular_acceleration = f"{float(match.group(1)):.2f} deg/s²"
                            data_buffer['angular_acceleration'] = float(match.group(1))
                    except Exception as e:
                        pass
                
                # 解析编码器数据
                elif line.startswith("ENC:"):
                    try:
                        # 格式: ENC:L:1234 R:5678
                        matches = re.findall(r'([LR]):(-?\d+)', line)
                        values = {k: int(v) for k, v in matches}
                        encoder_left = f"{values.get('L', 'N/A')}"
                        encoder_right = f"{values.get('R', 'N/A')}"
                        
                        data_buffer['encoder_left'] = values.get('L', 0)
                        data_buffer['encoder_right'] = values.get('R', 0)
                        data_buffer['timestamp'] = datetime.now().isoformat()
                    except Exception as e:
                        pass

            if current_command != last_command:
                arduino.write(current_command.encode())
                stdscr.move(1, 0)
                stdscr.clrtoeol()
                stdscr.addstr(1, 0, f"[>] Sent: {current_command}")
                stdscr.refresh()
                last_command = current_command

            stdscr.move(3, 0)
            stdscr.clrtoeol()
            stdscr.addstr(3, 0, f"[~] Distance: Front={distF}  Left={distL}  Right={distR}")
            
            stdscr.move(4, 0)
            stdscr.clrtoeol()
            stdscr.addstr(4, 0, f"[~] Angular Accel: {angular_acceleration}")
            
            stdscr.move(5, 0)
            stdscr.clrtoeol()
            stdscr.addstr(5, 0, f"[~] Encoders: Left={encoder_left}  Right={encoder_right}")
            
            stdscr.move(6, 0)
            stdscr.clrtoeol()
            stdscr.addstr(6, 0, f"[~] Data server: http://localhost:8081/data")
            stdscr.refresh()

            time.sleep(0.02)

        except KeyboardInterrupt:
            stdscr.addstr(7, 0, "[*] Interrupted by user.")
            stdscr.refresh()
            break


curses.wrapper(main)
arduino.close()
print("[*] Serial port closed.")
