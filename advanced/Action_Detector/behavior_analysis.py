import torch
import numpy as np
import cv2
import os
import sys
import time
import threading
import queue
from collections import deque
from urllib.parse import urlparse
import paho.mqtt.client as mqtt
import json

from LSTM import newLSTM
from dataset import CNNFeatureExtractor

# MQTT Configuration
MQTT_BROKER = "localhost"  
MQTT_PORT = 1883
MQTT_USERNAME = 'main'
MQTT_PASSWORD = 'sws3009-20'
MQTT_TOPIC_BEHAVIOR = 'robot/behavior'

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[*] MQTT connection successful for behavior analysis")
    else:
        print(f"[!] MQTT connection failed: {rc}")

class StreamBehaviorPredictor:
    def __init__(self, model_path='./best_action_classifier.pth'):
        """Real-time behavior predictor for video streams"""
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.sequence_length = 8
        self.action_names = {0: 'scratching', 1: 'eating', 2: 'playing'}
        
        # Frame buffer for sequence construction (stored in RAM)
        self.frame_buffer = deque(maxlen=50)  # Keep last 50 frames in memory
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Prediction control
        self.prediction_interval = 16  # Predict every 16 frames (~0.5s at 30fps)
        self.last_prediction_time = 0
        self.current_behavior = "unknown"
        self.current_confidence = 0.0
        
        print(f"Loading model from: {model_path}")
        self.model = newLSTM(576, 512, 3).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor('mobilenet_v3_small', self.device, True)
        
        print(f"Model loaded successfully! Training accuracy: {checkpoint['val_acc']:.1f}%")
        print(f"Using device: {self.device}")
        
        # Initialize MQTT
        self.init_mqtt()
    
    def init_mqtt(self):
        """Initialize MQTT client for behavior communication"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = on_mqtt_connect
            
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            
            print(f"[*] MQTT client initialized, connected to {MQTT_BROKER}:{MQTT_PORT}")
            return True
            
        except Exception as e:
            print(f"[!] MQTT initialization failed: {e}")
            return False
    
    def extract_features_from_frame(self, frame):
        """Extract CNN features from frame (BGR format from cv2)"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize using the same transforms as training
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor.model(input_tensor)
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy().flatten()
    
    def add_frame_to_buffer(self, frame):
        """Add frame to buffer and extract features"""
        features = self.extract_features_from_frame(frame)
        self.frame_buffer.append(features)
    
    def should_predict(self):
        """Check if we should make a prediction based on frame count and buffer size"""
        return (len(self.frame_buffer) >= self.sequence_length and 
                self.frame_count % self.prediction_interval == 0)
    
    def predict_from_buffer(self):
        """Predict behavior from current frame buffer"""
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        # Get the last sequence_length frames
        sequence_features = list(self.frame_buffer)[-self.sequence_length:]
        
        # Convert to tensor more efficiently
        sequence_array = np.array(sequence_features)
        sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(outputs).item()
        
        return {
            'action': self.action_names[predicted_class],
            'confidence': probabilities[predicted_class].item() * 100,
            'all_probabilities': {
                self.action_names[i]: probabilities[i].item() * 100 
                for i in range(len(self.action_names))
            },
            'timestamp': time.time()
        }
    
    def publish_behavior(self, prediction):
        """Publish behavior prediction via MQTT"""
        if prediction is None:
            return
        
        # Update current behavior
        self.current_behavior = prediction['action']
        self.current_confidence = prediction['confidence']
        
        # Prepare MQTT message
        behavior_msg = {
            'behavior': prediction['action'],
            'confidence': round(prediction['confidence'], 1),
            'timestamp': prediction['timestamp'],
            'probabilities': {k: round(v, 1) for k, v in prediction['all_probabilities'].items()}
        }
        
        # Publish to MQTT
        try:
            self.mqtt_client.publish(MQTT_TOPIC_BEHAVIOR, json.dumps(behavior_msg))
            print(f"🎯 Behavior: {prediction['action']} ({prediction['confidence']:.1f}%)")
        except Exception as e:
            print(f"MQTT publish error: {e}")
    
    def read_stream(self, stream_url):
        """Read frames from video stream and add to queue"""
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video stream {stream_url}")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Successfully connected to video stream: {stream_url}")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Cannot read video frame, attempting to reconnect...")
                time.sleep(1)
                continue
            
            # Drop old frames if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                # Rotate frame if needed (same as cat detection)
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.frame_queue.put(rotated_frame, timeout=0.1)
            except queue.Full:
                pass
        
        cap.release()
    
    def analyze_stream_behavior(self, stream_url):
        """
        Real-time behavior analysis from video stream
        
        Args:
            stream_url: URL of the video stream
        """
        # Validate URL
        parsed_url = urlparse(stream_url)
        if not parsed_url.scheme in ['http', 'https', 'rtmp', 'rtsp']:
            print("URL parsing error!")
            return
        
        self.is_running = True
        
        # Start stream reading thread
        stream_thread = threading.Thread(target=self.read_stream, args=(stream_url,))
        stream_thread.daemon = True
        stream_thread.start()
        
        print("Behavior analysis starting...")
        print("Press 'q' to quit")
        print("=" * 60)
        
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("Waiting for frame...")
                continue
            
            self.frame_count += 1
            self.fps_counter += 1
            
            # Update FPS counter
            current_time = time.time()
            if current_time - self.fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = current_time
            
            # Add frame to buffer for behavior analysis
            self.add_frame_to_buffer(frame)
            
            # Make prediction if conditions are met
            if self.should_predict():
                prediction = self.predict_from_buffer()
                if prediction:
                    self.publish_behavior(prediction)
            
            # Display frame with overlay info
            info_text = f'Frame: {self.frame_count} | FPS: {self.current_fps}'
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            behavior_text = f'Behavior: {self.current_behavior} ({self.current_confidence:.1f}%)'
            cv2.putText(frame, behavior_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, behavior_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            buffer_text = f'Buffer: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}'
            cv2.putText(frame, buffer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Cat Behavior Analysis', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.is_running = False
        cv2.destroyAllWindows()
        print("\nBehavior analysis stopped")

def main():
    stream_url = "http://192.168.148.103:8080/?action=stream"
    
    if len(sys.argv) > 1:
        stream_url = sys.argv[1]
    
    print(f"Cat Behavior Analysis - Real-time Stream Processing")
    print(f"Stream URL: {stream_url}")
    print(f"MQTT Topic: {MQTT_TOPIC_BEHAVIOR}")
    print("=" * 60)
    
    predictor = StreamBehaviorPredictor()
    
    try:
        predictor.analyze_stream_behavior(stream_url)
    except KeyboardInterrupt:
        print("\nUser interrupted program")
        predictor.is_running = False
    except Exception as e:
        print(f"Error occurred: {e}")
        predictor.is_running = False

if __name__ == "__main__":
    main()