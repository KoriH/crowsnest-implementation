from ultralytics import YOLO #For YOLO object detection model;
import supervision as sv #For tracking objects;
import numpy as np
import cv2 #For capturing and processing video frames;

# Initialize the model and tracker with error handling, if applicable;
try:
    model = YOLO("yolov8n.pt")#load the model;
    tracker = sv.ByteTrack()#Initialize the ByteTrack object tracker;
except Exception as e: #Catch and print any errors during initialization;
    print(f"Error initializing model or tracker: {e}")
    exit()# Kill the program if initialization fails;

# Initialize variables for tracking and analysis;
frame_id = 0 #Counter for the number of frames processed;
centroids = {}#Dictionary to store centroids of detected objects;
velocities = {}# Dictionary to store calculated velocities of tracked objects;
boxes = {}#Dictionary to store bounding boxes of detected objects;
frames = {}#Dictionary to track frames where each object was detected;
times = {}#Unused, but could be used to track timestamps of detections;

#function to compute the velocity of tracked objects;
def compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor):
    prev_x, prev_y = centroids.get(tracker_id, (centroid_x, centroid_y))
    frame_count = frames.get(tracker_id, 1)

    #Calculate displacement;
    dx = (centroid_x - prev_x) * scale_factor
    dy = (centroid_y - prev_y) * scale_factor
    
    velocity = np.sqrt(dx**2 + dy**2) / frame_count * 30
    velocities[tracker_id] = round(velocity, 1)

#Function to annotate frames with detected objects and their velocities;
def annotate_frame(frame, x1, y1, x2, y2, tracker_id, class_id):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)  # Convert coordinates to integers
    
    # Access the class name using the class_id from the model's 'names' list
    class_name = model.names[class_id]
    
    # Create the label text with the class name and tracker ID
    label = f"{class_name} {tracker_id}"
    
    # Draw the bounding box
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Determine label position
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_x = x1
    label_y = y1 - label_size[1] if y1 - label_size[1] > 10 else y1 + label_size[1]
    
    # Draw label background
    frame = cv2.rectangle(frame, (label_x, label_y-label_size[1]-2), (label_x+label_size[0], label_y+2), (0, 255, 0), cv2.FILLED)
    
    # Put the label text
    frame = cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

#Start video capture:
cap = cv2.VideoCapture(0)#0 for default camera;
annotated_frame = None#Initialize variable to store the annotated frame;
collision_times = {}#Unused, but could be used to track collision times;

# Main loop to process video frames
# Loop through all detections to annotate them
while cap.isOpened():
    success, frame = cap.read()

    if success:
        uncropped = frame.copy()
        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
            x1, y1, x2, y2 = box
            uncropped = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id, class_id)

        cv2.imshow("YOLOv8 Inference", uncropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


