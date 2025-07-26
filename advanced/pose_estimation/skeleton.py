# this is complement code for pose estimation adopting ultalytic package(since this package uses different kpts arrangement, I change it into the way how our datasets
# are arranged, so that the pic could really show the right skeleton of our cats)

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


model = YOLO('round200.pt')

image_path = 'test/image.png'
img = cv2.imread(image_path)

results = model(img)

custom_skeleton = [
    (0, 1),  
    (0, 2),  
    (1, 2), 
    (2, 3),   
    (3, 4),  
    (3, 5), 
    (5, 6),  
    (6, 7),
    (3, 8),
    (8, 9),
    (9, 10),
    (4, 11),
    (11, 12),
    (12, 13),
    (4, 14),
    (14, 15),
    (15, 16),
]


for r in results:
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        # 创建图像副本用于绘制
        annotated_img = img.copy()
        
        keypoints = r.keypoints.xy[0]  # 获取第一个检测到的关键点
        
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # 只绘制有效的关键点
                cv2.circle(annotated_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(annotated_img, str(i), (int(x+8), int(y-8)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        for connection in custom_skeleton:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(annotated_img, 
                            (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), 
                            (255, 0, 0), 2)
        
        cv2.imshow("Image with Skeleton", annotated_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()