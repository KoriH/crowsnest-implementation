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
def annotate_frame(frame, x1, y1, x2, y2, tracker_id, label):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)#Convert coordinates to integers;
    
    # Draw rectangle:
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Set up the position for the label:
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_x = x1
    label_y = y1 - label_size[1] if y1 - label_size[1] > 10 else y1 + label_size[1]
    # Draw rectangle for label:
    frame = cv2.rectangle(frame, (label_x, label_y-label_size[1]-2), (label_x+label_size[0], label_y+2), (0, 255, 0), cv2.FILLED)
    # Put the label text inside the rectangle:
    frame = cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

#Start video capture:
cap = cv2.VideoCapture(0)#0 for default camera;
annotated_frame = None#Initialize variable to store the annotated frame;
collision_times = {}#Unused, but could be used to track collision times;

# Main loop to process video frames
while cap.isOpened():
    success, frame = cap.read()  # Read a frame
    
    if success:  # If frame is read successfully
        frame_id += 1  # Increment frame counter
        uncropped = frame.copy()  # Make a copy of the frame
        
        # Perform object detection
        results = model(frame)[0]
        
        # Convert detections to a format compatible with the tracker
        detections = sv.Detections.from_ultralytics(results)
        
        # Update the tracker with the detections
        detections = tracker.update_with_detections(detections)
        
        # Loop through detections
        for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
            if class_id == 0:  # Assuming class_id 0 is for persons
                # Extract and adjust bounding box coordinates
                x1, y1, x2, y2 = box
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                
                # Additional read to ensure frame synchronization (may not be necessary)
                ret, frame = cap.read()
                
                if ret:  # If frame is read successfully
                    # Compute velocity of the tracked object
                    compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor=1)
                    # Update dictionaries with new data
                    centroids[tracker_id] = (centroid_x, centroid_y)
                    # Annotate the frame with detection data
                    label = f"Person {tracker_id}: {velocities[tracker_id]} m/s"
                    annotated_frame = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id, label)
                    frames[tracker_id] = frame_id
        
        # Display the annotated frame
        if annotated_frame is not None:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
