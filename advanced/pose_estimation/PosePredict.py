from ultralytics import YOLO 
import os 
import cv2
import numpy as np

print(os.getcwd())
model = YOLO('./round200.pt')

COCO_KEYPOINT_NAMES = [
    "left_eye",         # 0
    "right_eye",     # 1
    "nose",    # 2
    "neck",     # 3
    "root_of_tail",    # 4
    "left_shoulder", # 5
    "left_elbow", # 6
    "left_front_paw",   # 7
    "right_shoulder",  # 8
    "right_elbow",   # 9
    "right_front_paw",  # 10
    "left_hip",     # 11
    "left_knee",    # 12
    "left_back_paw",    # 13
    "left_back_paw",   # 14
    "right_hip",   # 15
    "right_back_paw"   # 16
]

def draw_keypoints_with_numbers(image_path, keypoints, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    kpts = keypoints.cpu().numpy()
    
    for i, (x, y, conf) in enumerate(kpts):
        if conf > 0.6: 
            
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            
          
            cv2.putText(img, str(i), (int(x+8), int(y-8)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if i < len(COCO_KEYPOINT_NAMES):
                cv2.putText(img, COCO_KEYPOINT_NAMES[i], (int(x+8), int(y+15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, img)
    print(f"带编号的关键点图片已保存至: {output_path}")

results = model('./test/image3.png')

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    
    if keypoints is not None:
        print("关键点坐标 (x, y, confidence):")
        kpts = keypoints[0, :, :].data
        print(kpts.shape)
        for i, (x, y, conf) in enumerate(kpts[0]):
            print(f"{i:2d}. {COCO_KEYPOINT_NAMES[i]:15s}: ({x:6.1f}, {y:6.1f}, {conf:4.2f})")
        
        draw_keypoints_with_numbers('./test/image3.png', keypoints[0, :, :].data[0], 'result_with_numbers.jpg')
    
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk