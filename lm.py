import cv2
import numpy as np
import math
from ultralytics import YOLO
from tabnanny import verbose

try:
    model = YOLO('yolov8n-face.pt')
except Exception as e:
    print(f"Error loading model. Make sure you are connected to internet on first run.\n{e}")
    exit()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)[0]

    # bounding box
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # key point(landmarks: eyes, nose, mouth)
    if results.keypoints:
        for kp_set in results.keypoints.data:
            for kp in kp_set:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow('Laughing Man - Phase 1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()