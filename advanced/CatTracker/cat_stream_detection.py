from ultralytics import YOLO
import cv2
import numpy as np
import sys
import time
import threading
import queue
from urllib.parse import urlparse
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "localhost"  
MQTT_PORT = 1883
MQTT_USERNAME = 'main'
MQTT_PASSWORD = 'sws3009-20'

MQTT_TOPIC_CENTRE = 'robot/object'

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[*] MQTT connection successful")
        # client.subscribe(MQTT_TOPIC_CENTRE)
    else:
        print(f"[!] MQTT connection failed: {rc}")

class StreamDetector:
    def __init__(self, model_path="./yolo11n.pt"):

        self.model = YOLO(model_path)
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        self.init_mqtt()


    def init_mqtt(self,):
        global mqtt_client
        
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = on_mqtt_connect
            # mqtt_client.on_disconnect = on_mqtt_disconnect
            # mqtt_client.on_message = on_mqtt_message
            
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            
            print(f"[*] MQTT客户端初始化完成，连接到 {MQTT_BROKER}:{MQTT_PORT}")
            return True
            
        except Exception as e:
            print(f"[!] MQTT初始化失败: {e}")
            return False

   
    def read_stream(self, stream_url):

        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频流 {stream_url}")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"成功连接到视频流: {stream_url}")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("警告：无法读取视频帧，尝试重新连接...")
                time.sleep(1)
                continue
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), timeout=0.1)
            except queue.Full:
                print("exception caught")
                pass
        
        cap.release()
    
    def detect_cats_in_stream(self, stream_url, save_detections=False):
        
        parsed_url = urlparse(stream_url)
        if not parsed_url.scheme in ['http', 'https', 'rtmp', 'rtsp']:
            print("Parsing URL error!")
        
        self.is_running = True
        
        stream_thread = threading.Thread(target=self.read_stream, args=(stream_url,))
        stream_thread.daemon = True
        stream_thread.start()
        
        print("Detection opening...")
        print("click 'q' to quit.")
        # print("按 's' 键保存当前帧")
        # print("按 'r' 键重置FPS计数")
        # print("=" * 60)
        
        saved_count = 0
        total_cats_detected = 0
        
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("Waiting for frame...")
                continue
            
            self.frame_count += 1
            self.fps_counter += 1
            
            current_time = time.time()
            if current_time - self.fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = current_time
            
            results = self.model.predict(source=frame, save=False, conf=0.25, verbose=False)
            
            cats_in_frame = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = box.conf[0].item()
                        class_name = self.model.names[class_id]
                        
                        if class_name.lower() == 'cat':
                            cats_in_frame += 1
                            
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                            
                            label = f'Cat {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                        (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
                            center_label = f'({center_x}, {center_y})'
                            center_label_size = cv2.getTextSize(center_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame, (center_x - center_label_size[0]//2 - 5, center_y - 25), 
                                        (center_x + center_label_size[0]//2 + 5, center_y - 5), (0, 0, 0), -1)
                            cv2.putText(frame, center_label, (center_x - center_label_size[0]//2, center_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            print(f"检测到猫 - 中心点: ({center_x}, {center_y}), 置信度: {confidence:.2f}")
                            msg = {
                                'x': center_x,
                                'y': center_y,
                            }
                        else:
                            msg = {
                                'x': -1000,
                                'y': -1000
                            }              
                        self.mqtt_client.publish(MQTT_TOPIC_CENTRE, json.dumps(msg))
            
            total_cats_detected += cats_in_frame
            
            info_text = f'Frame: {self.frame_count} | FPS: {self.current_fps} | Cats: {cats_in_frame}'
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            total_text = f'Total Cats Detected: {total_cats_detected}'
            cv2.putText(frame, total_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, total_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save frame, 'r' to reset FPS", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('HTTP Stream Cat Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # elif key == ord('s'):
            #     saved_count += 1
            #     timestamp = time.strftime("%Y%m%d_%H%M%S")
            #     filename = f'http_stream_cat_detected_{timestamp}_{saved_count}.jpg'
            #     cv2.imwrite(filename, frame)
            #     print(f"帧已保存为: {filename}")
            # elif key == ord('r'):
            #     self.fps_counter = 0
            #     self.fps_time = time.time()
            #     self.current_fps = 0
        
        self.is_running = False
        cv2.destroyAllWindows()
        
        # print(f"\n检测结束")
        # print(f"总处理帧数: {self.frame_count}")
        # print(f"总检测到猫: {total_cats_detected} 次")
        # print(f"保存的帧数: {saved_count}")
        return True

def main():
    stream_url = "http://192.168.148.103:8080/?action=stream"
    
    detector = StreamDetector()
    
    try:
        success = detector.detect_cats_in_stream(stream_url)
    
    except KeyboardInterrupt:
        print("\n用户中断程序")
        detector.is_running = False
    except Exception as e:
        print(f"发生错误: {e}")
        detector.is_running = False

if __name__ == "__main__":
    main() 