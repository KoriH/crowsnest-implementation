import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time


# Email Function (change the email address)
def send_email(subject, body):
    smtp_server = "smtp-mail.outlook.com"
    smtp_port = 587
    sender_email = "fulangxisi03@outlook.com"
    receiver_email = "francisyuxuanliu@yahoo.com"
    password = "sunsEf-zykso7-vyhwar"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


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



#### Email notification parameters
person_count_threshold = 3  
notification_cooldown = 300  # notification frequency limit
minimum_detection_duration = 5  # object persistence threshold in seconds
last_email_time = time.time() - notification_cooldown  # Allows sending immediately
person_detected_start_time = None

while cap.isOpened():
    success, frame = cap.read()

    if success:
        uncropped = frame.copy()
        results = model(frame)[0]

        # Convert detections to a format compatible with the tracker
        detections = sv.Detections.from_ultralytics(results)

        # Update the tracker with the detections
        detections = tracker.update_with_detections(detections)

        # Reset person count for each frame
        frame_person_count = 0

        # Loop through detections
        for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
            if class_id == 0:  # Assuming class_id 0 is for persons
                frame_person_count += 1
                x1, y1, x2, y2 = box
                # Annotate the frame with detection data
                label = f"Person {tracker_id}"
                uncropped = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id, label)

        # Person detection logic
        if frame_person_count > 0:
            if person_detected_start_time is None:
                person_detected_start_time = time.time()
            elif (time.time() - person_detected_start_time >= minimum_detection_duration and 
                  time.time() - last_email_time >= notification_cooldown):
                send_email("Alert: High Person Count", f"{frame_person_count} people detected.") # the email text
                last_email_time = time.time()
                person_detected_start_time = None  # Reset after sending email
        else:
            person_detected_start_time = None

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", uncropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
