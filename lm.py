from ctypes import resize

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

# load image for replacement
overlay_img = cv2.imread('images2.png', cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    print("Error: Could not load 'images2.png'.")
    exit()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)[0]

    # bounding box
    if results.boxes:
        # get original overlay dimensions to keep aspect ratio
        orig_h, orig_w = overlay_img.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # calculate the centre of the face
            centre_x = (x1 + x2) // 2
            centre_y = (y1 + y2) // 2

            # initialize the adjusted centre point
            adjusted_centre_x = centre_x
            adjusted_centre_y = centre_y

            # dynamic pose compensation (prevent misalignment)
            if results.keypoints is not None and len(results.keypoints.data) > i:
                kp_set = results.keypoints.data[i]
                if len(kp_set) >= 3:
                    nose_x = int(kp_set[2][0])
                    nose_y = int(kp_set[2][1])

                    # calculate the distance of the nose from the face box centre
                    nose_offset_x = nose_x - centre_x
                    nose_offset_y = nose_y - centre_y

                    # compensation factors
                    yaw_compensation = 1.3
                    pitch_compensation = 1.2

                    # reserve offset the overlay centre point based on nose position
                    adjusted_centre_x = centre_x - int(nose_offset_x * yaw_compensation)
                    adjusted_centre_y = centre_y - int(nose_offset_y * pitch_compensation)

            # get the actual face height, ignoring width changes, preventing overlay from shrinking
            face_h = y2 - y1

            #scale up the box size to cover the whole face
            scale_factor = 1.6

            # calculate target dimensions keeping original aspect ratio
            target_h = int(face_h * scale_factor)
            target_w = int(target_h * (orig_w / orig_h))

            # calculate new coordinates centered on the face
            new_x1 = centre_x - target_w // 2
            new_y1 = centre_y - target_h // 2
            new_x2 = new_x1 + target_w
            new_y2 = new_y1 + target_h

            # calculate safe boundaries for the frame (prevent crashing at screen edges)
            fx1, fy1 = max(0, new_x1), max(0, new_y1)
            fx2, fy2 = min(frame_w, new_x2), min(frame_h, new_y2)

            # calculate corresponding boundaries for the overlay image to crop it
            ox1, oy1 = fx1 - new_x1, fy1 - new_y1
            ox2, oy2 = target_w - (new_x2 - fx2), target_h - (new_y2 - fy2)
            
            # w = x2 - x1
            # h = y2 - y1

            if fx2 > fx1 and fy2 > fy1:
                resized_overlay = cv2.resize(overlay_img, (target_w, target_h))
                # crop overlay if it goes outside the camera frame
                cropped_overlay = resized_overlay[oy1:oy2, ox1:ox2]

                # check if the overlay image has an alpha channel(4 channels)
                if cropped_overlay.shape[2] == 4:
                    # extract the alpha mask and normalize it to 0.0 ~ 1.0
                    alpha_mask = cropped_overlay[:, :, 3] / 255.0
                    
                    # extract the BGR colour channels
                    overlay_bgr = cropped_overlay[:, :, :3]
                    
                    # extract the region of interest(roi) from the frame
                    roi = frame[fy1:fy2, fx1:fx2]

                    # blend the overlay and the roi using the alpha mask
                    for c in range(3):
                        roi[:, :, c] = (alpha_mask * overlay_bgr[:, :, c]) + (1.0 - alpha_mask) * roi[:, :, c]
                else:
                    # fallback if there is no alpha channel
                    frame[fy1:fy2, fx1:fx2] = cropped_overlay
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # key point(landmarks: eyes, nose, mouth)
    if results.keypoints:
        for kp_set in results.keypoints.data:
            for kp in kp_set:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow('Laughing Man - Phase 2: add pics', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()