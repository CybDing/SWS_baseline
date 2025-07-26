
# 这里的具体代码主要使用AI辅助编写，这里就是将模型对于猫识别结果得到的框加到视频输出界面上，从而可以在我们终端上可以获得bounding boxes
# original部分主要涉及单线程处理，处理稍微好像更佳容易卡顿，因此我后面主要直接修改为了线程处理，也就是将最新处理的结果画在视频界面上，从而使得视频尽可能没有卡顿

from flask import Flask, Response, url_for, request
import cv2
import datetime
from config import url
import time
from ultralytics import YOLO
import numpy as np
import json
import paho.mqtt.client as mqtt

app = Flask(__name__)

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = 'rasp'
MQTT_PASSWORD = 'sws3009-20-20'
MQTT_TOPIC_OBJECT = "robot/object-detection"

# Global MQTT client
mqtt_client = None
mqtt_connected = False

# Initialize YOLO models
detection_model = YOLO('./yolo11x.pt')
pose_model = None
try:
    pose_model = YOLO('../pose_estimation/round200.pt')
except:
    print("Pose model not found, using detection only")

# Cat pose keypoint names
COCO_KEYPOINT_NAMES = [
    "left_eye", "right_eye", "nose", "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "left_back_paw", "right_hip", "right_back_paw"
]

# MQTT Functions
def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    global mqtt_connected
    if rc == 0:
        print(f"[*] MQTT connected successfully")
        mqtt_connected = True
    else:
        print(f"[!] MQTT connection failed: {rc}")
        mqtt_connected = False

def on_mqtt_disconnect(client, userdata, rc):
    """MQTT disconnect callback"""
    global mqtt_connected
    print(f"[!] MQTT disconnected: {rc}")
    mqtt_connected = False

def init_mqtt():
    """Initialize MQTT client"""
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
    """Publish cat position to MQTT"""
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
    """Attempt to reconnect MQTT client"""
    global mqtt_client, mqtt_connected
    if mqtt_client and not mqtt_connected:
        try:
            mqtt_client.reconnect()
            print("[*] MQTT reconnection attempted")
        except Exception as e:
            print(f"[!] MQTT reconnection failed: {e}")

# Global variables for frame counting and bounding box tracking
frame_count = 0
last_cat_bbox = None
bbox_age = 0
MAX_BBOX_AGE = 3  # Keep previous bbox for up to 10 frames

# Confidence threshold for cat detection
CONFIDENCE_THRESHOLD = 0.5  # Only consider detections above this confidence as valid cats

# Global variables for keypoint tracking
last_keypoints = None
keypoints_age = 0
MAX_KEYPOINTS_AGE = 8  # Keep previous keypoints for longer

def process_and_generate_frames(use_pose=False):
    global frame_count, last_cat_bbox, bbox_age, last_keypoints, keypoints_age
    video_capture = cv2.VideoCapture(url)
    # Optimize video capture settings
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get latest frames
    video_capture.set(cv2.CAP_PROP_FPS, 20)        # Increased FPS for smoother video
    
    frame_count = 0  
    
    while True:
        if not video_capture.isOpened():
            print(f"Video source at {url} not opened. Retrying in 3 seconds...")
            time.sleep(3)
            video_capture.release()
            video_capture = cv2.VideoCapture(url)
            continue

        # Flush buffer to get latest frame
        video_capture.grab()  # Discard buffered frame
        success, frame = video_capture.read()
        if not success:
            print(f"Failed to grab frame from {url}. Reconnecting...")
            video_capture.release()
            time.sleep(1) 
            video_capture = cv2.VideoCapture(url)
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            video_capture.set(cv2.CAP_PROP_FPS, 20)
            continue

        frame_count += 1
        print(frame_count)
        
        # Process every 3rd frame instead of every 2nd for better FPS
        # But always show frames for smoother video
        process_detection = (frame_count % 8 == 0)
            
        timestamp = datetime.datetime.now() # use the timestemp the time we read the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if process_detection:
            processed_frame = process_cat_detection(frame, use_pose)
        else:
            # Use previous bounding box if available and recent
            processed_frame = draw_previous_bbox(frame, use_pose)
        
        cv2.putText(processed_frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

def process_cat_detection(frame, use_pose=False):
    global frame_count, last_cat_bbox, bbox_age, last_keypoints, keypoints_age
    """Process frame for cat detection with optional pose estimation"""
    cats_detected = 0
    cat_found = False
    print("process once")
   
    try:
        results = detection_model.predict(source=frame, save=False, conf=0.7, verbose=False)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    class_name = detection_model.names[class_id]
                    
                    if class_name.lower() == 'cat':
                        cats_detected += 1
                        
                        # Only consider high-confidence detections as valid cats
                        if confidence >= CONFIDENCE_THRESHOLD:
                            cat_found = True
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                           
                            # Calculate center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Store current bounding box for tracking
                            last_cat_bbox = {
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'confidence': confidence
                            }
                            bbox_age = 0
                            
                            # Publish cat position to MQTT
                            publish_cat_position(center_x, center_y, confidence)
                            
                            # Draw detection visualization
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                            
                            # Add detection label
                            label = f'Cat {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                        (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
                            # Add center coordinates
                            center_label = f'({center_x}, {center_y})'
                            center_label_size = cv2.getTextSize(center_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame, (center_x - center_label_size[0]//2 - 5, center_y - 25), 
                                        (center_x + center_label_size[0]//2 + 5, center_y - 5), (0, 0, 0), -1)
                            cv2.putText(frame, center_label, (center_x - center_label_size[0]//2, center_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # If pose estimation is enabled and cat is small (box area < threshold)
                            # Only run pose estimation on every 4th processed frame to save computation
                            box_area = (x2 - x1) * (y2 - y1)
                            if use_pose and pose_model:
                                frame = add_pose_estimation(frame, x1, y1, x2, y2)
                        else:
                            # Low confidence detection - draw it differently to indicate uncertainty
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Draw dashed/dim bounding box for low confidence
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 100), 2)  # Dim yellow
                            
                            # Add low confidence label
                            label = f'Cat? {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                        (x1 + label_size[0] + 10, y1), (0, 100, 100), -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 在发送没有检测到猫的指令前给予一定延时，在连续三个循环中没有检测到猫才会发送-1000代码表示没有检测到猫 
        if not cat_found:
            bbox_age += 1
            if bbox_age > MAX_BBOX_AGE:
                publish_cat_position(-1000, -1000, 0.0)
 
        # Add detection info
        info_text = f'Cats detected: {cats_detected}'
        if use_pose:
            info_text += ' | Pose: ON'
        mqtt_status = 'MQTT: ON' if mqtt_connected else 'MQTT: OFF'
        info_text += f' | {mqtt_status}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
    except Exception as e:
        error_text = f'Detection error: {str(e)[:50]}'
        cv2.putText(frame, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def draw_previous_bbox(frame, use_pose=False):
    """Draw previous bounding box if available and recent"""
    global last_cat_bbox, bbox_age
    
    if last_cat_bbox and bbox_age <= MAX_BBOX_AGE:
        x1, y1, x2, y2 = last_cat_bbox['bbox']
        center_x, center_y = last_cat_bbox['center']
        confidence = last_cat_bbox['confidence']
        
        # Draw bounding box with reduced opacity (dashed style)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0), 2)  # Dimmer green
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 150), -1)  # Dimmer red
        
        # Add tracking label
        label = f'Tracking {confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                    (x1 + label_size[0] + 10, y1), (0, 150, 0), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        info_text = f'Cats detected: 1 (tracking)'
        if use_pose:
            info_text += ' | Pose: ON'
        mqtt_status = 'MQTT: ON' if mqtt_connected else 'MQTT: OFF'
        info_text += f' | {mqtt_status}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    else:
        info_text = 'Cats detected: 0'
        if use_pose:
            info_text += ' | Pose: ON'
        mqtt_status = 'MQTT: ON' if mqtt_connected else 'MQTT: OFF'
        info_text += f' | {mqtt_status}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    return frame
def add_pose_estimation(frame, x1, y1, x2, y2):

    """对猫进行姿态检测 这一个实际没有用到实际最后的demo之中 一个原因是pose模型准确度在非清晰图片上其实有时候准确度没有很高 其次也没有对pose数据进行训练对应的LSTM模型 因此这里最后没有
    使用，但是实际上可以在画面上将这些关键点给标记出来"""
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
                
                # Update stored keypoints if we found new ones
                if keypoints_found:
                    last_keypoints = valid_keypoints
                    keypoints_age = 0
                
                # Draw current keypoints
                for i, x, y, conf in valid_keypoints:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 255), -1)
                    if i < len(COCO_KEYPOINT_NAMES):
                        cv2.putText(frame, str(i), (int(x) + 3, int(y) - 3), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # Show pose label
                if keypoints_in_bbox > 0:
                    cv2.putText(frame, f'Pose({keypoints_in_bbox})', (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # If no new keypoints found, use stored ones if they're still fresh
        if not keypoints_found and last_keypoints and keypoints_age <= MAX_KEYPOINTS_AGE:
            keypoints_age += 1
            # Draw stored keypoints with reduced opacity
            for i, x, y, conf in last_keypoints:
                cv2.circle(frame, (int(x), int(y)), 2, (200, 0, 200), -1)  # Dimmer color
                if i < len(COCO_KEYPOINT_NAMES):
                    cv2.putText(frame, str(i), (int(x) + 3, int(y) - 3), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 0), 1)  # Dimmer text
            
            cv2.putText(frame, f'Pose(tracking)', (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
        
        # Reset stored keypoints if they're too old
        if keypoints_age > MAX_KEYPOINTS_AGE:
            last_keypoints = None
    
    except Exception as e:
        print(f"Find pose model exception:{e}")
    
    return frame

# 在这个url端口下面可以输出我们处理后的视频画面，将输出的画面基本处理后可以在控制界面进行展现
@app.route('/video_feed')
def video_feed():
    use_pose = request.args.get('pose', 'false').lower() == 'true'
    return Response(process_and_generate_frames(use_pose),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 这个是我们视频监控的主界面，包括一些基本的文字描述（使用html渲染，加上一个中央的可以暂时视频的一个视频框）
@app.route('/')
def index():
    """Main page for video feed."""
    return f"""
    <html>
        <head>
            <title>Cat Detection Video Stream</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .controls {{ margin: 20px 0; }}
                button {{ padding: 10px 20px; margin: 5px; }}
                #video {{ border: 2px solid #333; }}
            </style>
        </head>
        <body>
            <h1>Cat Detection & Tracking Stream</h1>
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
    
    print("[*] Starting Flask server...")
    app.run(host='0.0.0.0', port=5200, debug=True)