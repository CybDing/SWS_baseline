from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')  

input_folder = "./raw_pictures/pallas"
output_folder = "./cropped_pictures"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    img_path = os.path.join(input_folder, fname)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    results = model(img)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id]
            if label != "cat":
                continue  # 只处理cat

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            print(f"cat center: ({cx}, {cy}), type: {type(cx)}, x1y1x2y2 dtype: {box.xyxy[0].cpu().numpy().dtype}")

            # 计算最大正方形半边长
            half_size = min(cx, w - cx, cy, h - cy)
            left = cx - half_size
            right = cx + half_size
            top = cy - half_size
            bottom = cy + half_size

            # 防止边界越界
            left = max(left, 0)
            right = min(right, w)
            top = max(top, 0)
            bottom = min(bottom, h)

            crop = img[top:bottom, left:right]
            save_name = f"{os.path.splitext(fname)[0]}_cat_{cx}_{cy}_maxsquare.jpg"
            cv2.imwrite(os.path.join(output_folder, save_name), crop)

print("Done.")