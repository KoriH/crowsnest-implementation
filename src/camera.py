from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
import time

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
vidObj = cv2.VideoCapture('topdown.mp4') 

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}
times = {}


# for collision velocity
def compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor):

    prev_x, prev_y = centroids.get(tracker_id, (centroid_x, centroid_y))
    frame_count = frames.get(tracker_id, 1)
    

    dx = (centroid_x - prev_x) * scale_factor
    dy = (centroid_y - prev_y) * scale_factor
    

    velocity = np.sqrt(dx**2 + dy**2) / frame_count * 30
    velocities[tracker_id] = round(velocity, 1)

# bounding box for models
def annotate_frame(frame, x1, y1, x2, y2, tracker_id):
    # Round bounding box coordinates
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    
    # Draw bounding box and velocity label
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    frame = cv2.putText(frame, f"{velocities[tracker_id]}", (x2+7, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return frame

# red frame for collision
def collision_frame(frame, tracker_id, other_tracker_id):
    # Threshold for collision detection (Major)
    if velocities[tracker_id] > 10 or velocities[other_tracker_id] > 10:
        box1 = boxes[tracker_id]
        box2 = boxes[other_tracker_id]
        
        x1_1, y1_1, x2_1, y2_1 = map(int, box1)
        x1_2, y1_2, x2_2, y2_2 = map(int, box2)

        label_x1 = int((x1_1 + x2_1) / 2)
        label_y1 = int((y1_1 + y2_1) / 2)

        label_x2 = int((x1_2 + x2_2) / 2)
        label_y2 = int((y1_2 + y2_2) / 2)

        frame = cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 2)

        frame = cv2.putText(frame, "Major Collision Detected!", (label_x1, label_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = cv2.putText(frame, "Major Collision Detected!", (label_x2, label_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    return frame


cap = cv2.VideoCapture(0)  
annotated_frame = None 
collision_times = {}

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        frame_id += 1
        uncropped = frame.copy()
        frame = frame[220:530, :]
        
        # Perform inference on the frame
        results = model(frame)[0]
        
        # Format detections object
        detections = sv.Detections.from_ultralytics(results)
        
        detections = tracker.update_with_detections(detections)
        
        for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
            if class_id == 0:
                x1, y1, x2, y2 = box 
                y1, y2 = y1 + 220, y2 + 220
                boxes[tracker_id] = (x1, y1, x2, y2)
                
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                ret, frame = cap.read()

                if ret:
                    frame_height, frame_width = frame.shape[:2]
                    compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor=1)
                    centroids[tracker_id] = (centroid_x, centroid_y)
                    annotated_frame = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id)
                    frames[tracker_id] = frame_id
            

        # Display the annotated frame
        if annotated_frame is not None:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()