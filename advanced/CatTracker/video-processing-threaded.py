from flask import Flask, Response, url_for, request
import cv2
import datetime
from config import url
import time
from ultralytics import YOLO
import numpy as np
import json
import paho.mqtt.client as mqtt
import threading
import queue

app = Flask(__name__)

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = 'main'
MQTT_PASSWORD = 'sws3009-20'
MQTT_TOPIC_OBJECT = "robot/object-detection"

mqtt_client = None
mqtt_connected = False

detection_model = YOLO('./yolo11x.pt')
pose_model = None
try:
    pose_model = YOLO('../pose_estimation/round200.pt')
except:
    print("Pose model not found, using detection only")

COCO_KEYPOINT_NAMES = [
    "left_eye", "right_eye", "nose", "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "left_back_paw", "right_hip", "right_back_paw"
]

def on_mqtt_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print(f"[*] MQTT connected successfully")
        mqtt_connected = True
    else:
        print(f"[!] MQTT connection failed: {rc}")
        mqtt_connected = False

def on_mqtt_disconnect(client, userdata, rc):
    global mqtt_connected
    print(f"[!] MQTT disconnected: {rc}")
    mqtt_connected = False

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_disconnect = on_mqtt_disconnect
        
        if MQTT_USERNAME and MQTT_PASSWORD:
            mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[*] MQTT client initialized, connecting to {MQTT_BROKER}:{MQTT_PORT}")
        return True
    except Exception as e:
        print(f"[!] MQTT initialization failed: {e}")
        return False

def publish_cat_position(x, y, confidence=0.0):
    global mqtt_client, mqtt_connected
    if mqtt_client and mqtt_connected:
        try:
            message = {
                "x": int(x),
                "y": int(y),
                "confidence": float(confidence),
                "timestamp": datetime.datetime.now().isoformat()
            }
            result = mqtt_client.publish(MQTT_TOPIC_OBJECT, json.dumps(message))
            if result.rc != 0:
                print(f"[!] MQTT publish failed with code: {result.rc}")
                return False
            return True
        except Exception as e:
            print(f"[!] Failed to publish position: {e}")
            try_mqtt_reconnect()
    return False

def try_mqtt_reconnect():
    global mqtt_client, mqtt_connected
    if mqtt_client and not mqtt_connected:
        try:
            mqtt_client.reconnect()
            print("[*] MQTT reconnection attempted")
        except Exception as e:
            print(f"[!] MQTT reconnection failed: {e}")

frame_count = 0
last_cat_bbox = None
bbox_age = 0
MAX_BBOX_AGE = 1
CONFIDENCE_THRESHOLD = 0.6
last_keypoints = None
keypoints_age = 0
MAX_KEYPOINTS_AGE = 8

frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=20)
processing_thread = None
processing_active = False

searching_mode = False
cat_found_notification_sent = False

def processing_worker(use_pose=False):
    """Background thread for processing frames"""
    global processing_active, frame_count
    processing_active = True
    
    while processing_active:
        try:
            frame_data = frame_queue.get(timeout=1.0)
            if frame_data is None:  # Shutdown signal
                break
                
            frame, frame_num = frame_data
            
            if frame_num % 4 == 0:
                process_cat_detection(frame.copy(), use_pose)
                result_queue.put({
                    'frame_num': frame_num,
                    'processed': True,
                    'overlay_data': get_overlay_data()
                })
            
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing worker error: {e}")
            continue
    
    print("Processing worker stopped")

def get_overlay_data():
    """Extract current overlay data for display thread"""
    global last_cat_bbox, bbox_age, mqtt_connected
    return {
        'last_cat_bbox': last_cat_bbox.copy() if last_cat_bbox else None,
        'bbox_age': bbox_age,
        'mqtt_connected': mqtt_connected
    }

def apply_overlay_data(frame, overlay_data, use_pose=False):
    """Apply overlay data to frame for display"""
    if overlay_data and overlay_data['last_cat_bbox'] and overlay_data['bbox_age'] <= MAX_BBOX_AGE:
        bbox_data = overlay_data['last_cat_bbox']
        x1, y1, x2, y2 = bbox_data['bbox']
        center_x, center_y = bbox_data['center']
        confidence = bbox_data['confidence']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
        
        # Add labels
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
        
        info_text = 'Cats detected: 1'
    else:
        info_text = 'Cats detected: 0'
    
    if use_pose:
        info_text += ' | Pose: ON'
    mqtt_status = 'MQTT: ON' if overlay_data and overlay_data['mqtt_connected'] else 'MQTT: OFF'
    info_text += f' | {mqtt_status}'
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    return frame

def process_and_generate_frames(use_pose=False):
    global frame_count, processing_thread, processing_active
    
    # Start processing thread if not already running
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=processing_worker, args=(use_pose,), daemon=True)
        processing_thread.start()
        print("Processing thread started")
    
    video_capture = cv2.VideoCapture(url)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FPS, 25)  # Optimized FPS for performance
    
    frame_count = 0
    last_overlay_data = None
    
    try:
        while True:
            if not video_capture.isOpened():
                print(f"Video source at {url} not opened. Retrying in 3 seconds...")
                time.sleep(3)
                video_capture.release()
                video_capture = cv2.VideoCapture(url)
                continue

            video_capture.grab()
            success, frame = video_capture.read()
            if not success:
                print(f"Failed to grab frame from {url}. Reconnecting...")
                video_capture.release()
                time.sleep(1)
                video_capture = cv2.VideoCapture(url)
                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                video_capture.set(cv2.CAP_PROP_FPS, 30)
                continue

            frame_count += 1
            timestamp = datetime.datetime.now()
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Send frame to processing thread (non-blocking with aggressive dropping)
            try:
                # Clear old frames if queue is getting full
                if frame_queue.qsize() >= 3:
                    try:
                        frame_queue.get_nowait()  # Drop oldest frame
                    except queue.Empty:
                        pass
                frame_queue.put_nowait((frame.copy(), frame_count))
            except queue.Full:
                # If still full, clear queue and add current frame
                try:
                    while not frame_queue.empty():
                        frame_queue.get_nowait()
                    frame_queue.put_nowait((frame.copy(), frame_count))
                except (queue.Empty, queue.Full):
                    pass
            
            # Check for processing results
            try:
                while True:
                    result = result_queue.get_nowait()
                    last_overlay_data = result['overlay_data']
                    result_queue.task_done()
            except queue.Empty:
                pass
            
            display_frame = apply_overlay_data(frame, last_overlay_data, use_pose)
            
            # Add timestamp
            cv2.putText(display_frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

            (flag, encodedImage) = cv2.imencode(".jpg", display_frame)
            if not flag:
                continue

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
                  
    finally:
        processing_active = False
        try:
            frame_queue.put_nowait(None)  # Shutdown signal
        except queue.Full:
            pass
        video_capture.release()
        print("Video capture cleanup completed")

def process_cat_detection(frame, use_pose=False):
    global frame_count, last_cat_bbox, bbox_age, last_keypoints, keypoints_age
    """Process frame for cat detection with optional pose estimation"""
    cats_detected = 0
    cat_found = False
   
    try:
        results = detection_model.predict(source=frame, save=False, conf=0.5, verbose=False, device='cpu')
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    class_name = detection_model.names[class_id]
                    
                    if class_name.lower() == 'cat':
                        cats_detected += 1
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            cat_found = True
                            
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                           
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Store current bounding box for tracking
                            last_cat_bbox = {
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'confidence': confidence
                            }
                            bbox_age = 0
                            
                            publish_cat_position(center_x, center_y, confidence)
                            
                            if use_pose and pose_model:
                                box_area = (x2 - x1) * (y2 - y1)
                                add_pose_estimation(frame, x1, y1, x2, y2)
        
        if not cat_found:
            bbox_age += 1
            if bbox_age > MAX_BBOX_AGE:
                publish_cat_position(-1000, -1000, 0.0)
        
    except Exception as e:
        print(f"Detection error: {e}")
    
    return frame

def add_pose_estimation(frame, x1, y1, x2, y2):
    """Add pose estimation for detected cat region"""
    global last_keypoints, keypoints_age
    
    try:
        pose_results = pose_model.predict(source=frame, save=False, conf=0.7, verbose=False, classes=[15])
        
        keypoints_found = False
        for result in pose_results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                keypoints = result.keypoints.data[0]
                
                valid_keypoints = []
                keypoints_in_bbox = 0
                
                for i, keypoint in enumerate(keypoints):
                    x, y, conf = keypoint.cpu().numpy()
                    if (x > 0 and y > 0 and conf > 0.2 and 
                        x1 <= x <= x2 and y1 <= y <= y2):
                        valid_keypoints.append((i, x, y, conf))
                        keypoints_in_bbox += 1
                        keypoints_found = True
                
                if keypoints_found:
                    last_keypoints = valid_keypoints
                    keypoints_age = 0
        
        if keypoints_age > MAX_KEYPOINTS_AGE:
            last_keypoints = None
    
    except Exception as e:
        print(f"Pose model exception: {e}")
    
    return frame

@app.route('/video_feed')
def video_feed():
    use_pose = request.args.get('pose', 'false').lower() == 'true'
    return Response(process_and_generate_frames(use_pose),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Main page for video feed."""
    return f"""
    <html>
        <head>
            <title>Cat Detection Video Stream (Threaded)</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .controls {{ margin: 20px 0; }}
                button {{ padding: 10px 20px; margin: 5px; }}
                #video {{ border: 2px solid #333; }}
                .info {{ margin: 10px 0; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Cat Detection & Tracking Stream (Threaded Version)</h1>
            <div class="info">
                <p><strong>Performance Optimized:</strong> Video display runs at 30fps while detection runs every 8th frame in background thread</p>
            </div>
            <div class="controls">
                <button onclick="togglePose()">Toggle Pose Estimation</button>
                <span id="status">Pose: OFF</span>
            </div>
            <img id="video" src="{url_for('video_feed')}" width="800">
            
            <script>
                let poseEnabled = false;
                
                function togglePose() {{
                    poseEnabled = !poseEnabled;
                    const video = document.getElementById('video');
                    const status = document.getElementById('status');
                    
                    video.src = '{url_for('video_feed')}?pose=' + poseEnabled;
                    status.textContent = 'Pose: ' + (poseEnabled ? 'ON' : 'OFF');
                }}
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("[*] Initializing MQTT connection...")
    init_mqtt()
    
    print("[*] Starting Flask server with threaded video processing...")
    app.run(host='0.0.0.0', port=5200, debug=True)