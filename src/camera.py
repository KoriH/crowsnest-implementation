from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

model = YOLO("best.pt")
tracker = sv.ByteTrack()

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}
times = {}



def compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor):

    prev_x, prev_y = centroids.get(tracker_id, (centroid_x, centroid_y))
    frame_count = frames.get(tracker_id, 1)
    

    dx = (centroid_x - prev_x) * scale_factor
    dy = (centroid_y - prev_y) * scale_factor
    

    velocity = np.sqrt(dx**2 + dy**2) / frame_count * 30
    velocities[tracker_id] = round(velocity, 1)

def annotate_frame(frame, x1, y1, x2, y2, tracker_id):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    frame = cv2.putText(frame, f"{velocities[tracker_id]}", (x2+7, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
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
        
        results = model(frame)[0]
        
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
            

        if annotated_frame is not None:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()