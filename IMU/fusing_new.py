import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt
from collections import deque
import socket
import json
import requests
from datetime import datetime

SERVER_IP = '192.168.0.1'
class IMU:
    def __init__(self, server_host=SERVER_IP, server_port=8081):
        self.topic = 'IMU'
        self.YAWangle = 0
        self.past_EncoderValue = deque([0.0] * 15, maxlen=15)
        self.past_YAW_angular_acc = deque([0.0] * 15, maxlen=15)
        self.encoder_gain = 1
        self.speed = 0
        self.turning = False
        self.positions = [(0, 0), (0, 0)]  
        self.timestep = 0
        self.time = None
        
        self.filter_order = 2
        self.cutoff_freq = 0.1
        
        # 服务器连接相关
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.connected = False
        self.time_offset = 0  # 与服务器时间差
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.trajectory_lines = [] 
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_title('Vehicle Trajectory')
        
    def connect(self):
        '''
        连接到服务器，建立socket连接并同步时间
        '''
        try:
            # 建立socket连接
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"[*] Connected to server {self.server_host}:{self.server_port}")
            
            # 时间同步
            self._sync_time()
            
        except Exception as e:
            print(f"[!] Failed to connect to server: {e}")
            self.connected = False
            self.socket = None
            
    def _sync_time(self):
        '''
        与服务器同步时间，计算时间偏移
        '''
        try:
            # 使用HTTP GET请求获取服务器时间戳
            url = f"http://{self.server_host}:{self.server_port}/data"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                server_timestamp = data.get('timestamp', '')
                
                if server_timestamp:
                    # 解析服务器时间戳
                    server_time = datetime.fromisoformat(server_timestamp.replace('Z', '+00:00'))
                    local_time = datetime.now()
                    
                    # 计算时间偏移（秒）
                    self.time_offset = (server_time - local_time).total_seconds()
                    print(f"[*] Time synchronized. Offset: {self.time_offset:.3f} seconds")
                else:
                    print("[!] No timestamp received from server")
                    self.time_offset = 0
            else:
                print(f"[!] Failed to get server time: HTTP {response.status_code}")
                self.time_offset = 0
                
        except Exception as e:
            print(f"[!] Time sync failed: {e}")
            self.time_offset = 0
    
    def get_data(self):
        '''
        从服务器获取传感器数据
        '''
        if not self.connected:
            print("[!] Not connected to server")
            return None, None, None, None
            
        try:
            # 使用HTTP GET请求获取数据
            url = f"http://{self.server_host}:{self.server_port}/data"
            response = requests.get(url, timeout=1)
            
            if response.status_code == 200:
                data = response.json()
                
                # 提取数据
                angular_acc = data.get('angular_acceleration', 0)
                encoder_left = data.get('encoder_left', 0)
                encoder_right = data.get('encoder_right', 0)
                timestamp_str = data.get('timestamp', '')
                
                # 处理时间戳
                timestamp = None
                if timestamp_str:
                    try:
                        server_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # 调整为本地时间
                        local_time = server_time.timestamp() - self.time_offset
                        timestamp = local_time
                    except:
                        timestamp = time.time()
                else:
                    timestamp = time.time()
                
                return angular_acc, encoder_left, encoder_right, timestamp
            else:
                print(f"[!] Failed to get data: HTTP {response.status_code}")
                return None, None, None, None
                
        except Exception as e:
            print(f"[!] Error getting data: {e}")
            return None, None, None, None
    
    def disconnect(self):
        '''
        断开服务器连接
        '''
        if self.socket:
            try:
                self.socket.close()
                print("[*] Disconnected from server")
            except:
                pass
            finally:
                self.socket = None
                self.connected = False

    def _filter_angle_acc(self):

        if self.turning == False:
            return
            
        if len(self.past_YAW_angular_acc) < 3:
            return
        
        data = np.array(self.past_YAW_angular_acc)
        
        nyquist = 0.5 / self.timestep if self.timestep > 0 else 1
        normal_cutoff = min(self.cutoff_freq / nyquist, 0.99)
        b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = filtfilt(b, a, data)
        
        self.YAWangle += filtered_data[-1] * self.timestep

    def _filter_encoder(self, mode=3):
        '''
        Weighted average
        '''
        if self.turning or len(self.past_EncoderValue) == 0:
            return 0
        
        mode = min(mode, len(self.past_EncoderValue))
        
        recent_values = list(self.past_EncoderValue)[-mode:]
        weights = np.exp(np.arange(len(recent_values)) * 0.1)
        weighted_sum = np.sum(np.array(recent_values) * weights)
        weight_sum = np.sum(weights)
        
        return weighted_sum / weight_sum * self.encoder_gain

    def _update_position(self):

        if self.turning:
            return

        self.speed = self._filter_encoder()
        
        if self.speed == 0:
            return
            
        displacement = self.speed * self.timestep
        
        R = np.array([
            [np.cos(self.YAWangle), -np.sin(self.YAWangle)],
            [np.sin(self.YAWangle), np.cos(self.YAWangle)]
        ])
        
        self.positions[0] = self.positions[1]  # 上一个位置
        movement = displacement * R[:, 0] 
        self.positions[1] = (
            self.positions[0][0] + movement[0],
            self.positions[0][1] + movement[1]
        )

    def _draw_cur_pos(self):

        if self.positions[0] == self.positions[1]:
            return
            
        x_coords = [self.positions[0][0], self.positions[1][0]]
        y_coords = [self.positions[0][1], self.positions[1][1]]
        
        line, = self.ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        self.trajectory_lines.append(line)
        
        self.ax.plot(self.positions[1][0], self.positions[1][1], 'ro', markersize=6)
        
        all_x = [pos[0] for pos in self.positions]
        all_y = [pos[1] for pos in self.positions]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        x_range = max(x_max - x_min, 2)
        y_range = max(y_max - y_min, 2)
        
        self.ax.set_xlim(x_min - x_range*0.1, x_max + x_range*0.1)
        self.ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
        
        plt.pause(0.01)  

    def receive_all(self, angular_acc, LE, RE, timestamp, real_world=True):
        '''
        Read sensor_data
        '''
        # 在实际应用中，这里应该是从Rasp接收数据
        
        # simulated_angular_acc = np.random.normal(0, 0.05)
        # self.past_YAW_angular_acc.append(simulated_angular_acc)
        
        # simulated_encoder = np.random.normal(1.0, 0.1) if not self.turning else 0
        # self.past_EncoderValue.append(max(0, simulated_encoder))
        
        # if hasattr(self, 'step_count'):
        #     self.step_count += 1
        # else:
        #     self.step_count = 0
            
        # if self.step_count % 50 == 0:
        #     self.turning = not self.turning
        #     print(f"{'开始转弯' if self.turning else '结束转弯'}")
        if real_world:
            # 从服务器获取实际数据
            if self.connected:
                angular_acc, LE, RE, timestamp = self.get_data()
            
            if timestamp is None:
                print("No timestamp received!")
                timestamp = time.time()
                
            
            if angular_acc is not None:
                self.past_YAW_angular_acc.append(angular_acc)
            else: 
                print("No angular_acc received!, acc is set to 0")
                self.past_YAW_angular_acc.append(0)

            if LE is not None and RE is not None:
                # 使用左右编码器的平均值
                self.past_EncoderValue.append((LE + RE) / 2.0)
            elif LE is not None and RE is None:
                print("Right encoder not received, use Left encoder instead!")
                self.past_EncoderValue.append(LE)
            elif RE is not None and LE is None:
                print("Left encoder not received! Use right encoder instead!")
                self.past_EncoderValue.append(RE)   
            else:
                print("No encoder received, encoder value is set to zero!")
                self.past_EncoderValue.append(0)

        # 更新时间步长
        cur_time = timestamp if timestamp else time.time()
        if self.time is not None:
            self.timestep = cur_time - self.time
        self.time = cur_time

    def restart(self, test=False, connect_server=True):
        self.YAWangle = 0
        self.past_EncoderValue = deque([0.0] * 15, maxlen=15)
        self.past_YAW_angular_acc = deque([0.0] * 15, maxlen=15)
        self.time = time.time()
        self.speed = 0
        self.turning = False
        self.positions = [(0, 0), (0, 0)]
        
        # 连接服务器
        if connect_server and not test:
            self.connect()
        
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_title('Vehicle Trajectory')

        try:
            while True:
                if test:
                    # 测试模式，使用模拟数据
                    self.receive_all(None, None, None, None, real_world=False)
                else:
                    # 实际模式，从服务器获取数据
                    self.receive_all(None, None, None, None, real_world=True)
                
                self._filter_angle_acc()
                self._update_position()
                self._draw_cur_pos()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Program Stopped!")
            self.disconnect()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    imu = IMU()
    # 使用真实服务器数据
    imu.restart(test=False, connect_server=True)
    
    # 或者使用测试模式
    # imu.restart(test=True, connect_server=False)