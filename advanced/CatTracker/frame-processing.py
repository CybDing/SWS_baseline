import time
import cv2
from config import url

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
CurrentTime = time.time()
index = 0
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    index = index + 1 
    if index % 10 == 0 and index != 0:
        print("VideoFPS", 10/(time.time()-CurrentTime))
        CurrentTime = time.time()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
