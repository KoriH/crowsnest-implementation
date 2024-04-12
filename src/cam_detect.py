import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from datetime import datetime, timedelta

# Email Function (change the email address)
def send_email(subject, body):
    smtp_server = "smtp-mail.outlook.com"
    smtp_port = 587
    sender_email = "yvrhackathonmindbenders1@outlook.com"
    receiver_email = "francisyuxuanliu@yahoo.com"
    password = "wukdud-4howmy-tykqIb"

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


# Initialize the model and tracker with error handling
try:
    model = YOLO("yolov8n.pt")  # Load the model
    tracker = sv.ByteTrack()  # Initialize the ByteTrack object tracker
except Exception as e:
    print(f"Error initializing model or tracker: {e}")
    exit()  # Kill the program if initialization fails

# List of trash classes (all in lower case for consistency)
trash_classes = [
    'aluminium foil', 'bottle cap', 'bottle', 'broken glass', 'can',
    'carton', 'cigarette', 'cup', 'lid', 'other litter',
    'other plastic', 'paper', 'plastic bag - wrapper', 'plastic container', 'pop tab',
    'straw', 'styrofoam piece', 'unlabeled litter'
]

# Define specific classes for staff and crews
staff_and_crew_classes = ['air canada crew', 'airport staff', 'westjet crew']

# List of unattended item classes
unattended_item_classes = {26: 'handbag', 28: 'suitcase', 31: 'snowboard', 36: 'skateboard', 37: 'surfboard', 63: 'laptop', 67: 'cell phone'}

def check_proximity(person_boxes, item_box, threshold=100):
    """Check if any person is within a certain distance of a suitcase."""
    x1_s, y1_s, x2_s, y2_s = item_box
    suitcase_center = ((x1_s + x2_s) // 2, (y1_s + y2_s) // 2)

    for box in person_boxes:
        x1_p, y1_p, x2_p, y2_p = box
        person_center = ((x1_p + x2_p) // 2, (y1_p + y2_p) // 2)

        # Calculate Euclidean distance between the centers of person and suitcase
        distance = np.sqrt((person_center[0] - suitcase_center[0]) ** 2 + (person_center[1] - suitcase_center[1]) ** 2)
        if distance < threshold:
            return True
    return False

#Function to annotate frames with detected objects and their velocities;
def annotate_frame(frame, x1, y1, x2, y2, tracker_id, class_id, unattended=False):
    """Annotate the frame based on object detection."""
    class_name = model.names[class_id].lower()

    # Determine the label based on class
    if class_name in trash_classes:
        label = f"{class_name}_Trash {tracker_id}"
        color = (0, 255, 0)  # Green color
    elif class_id == 0:  # Special handling for passenger
        label = f"passenger {tracker_id}"
        color = (0, 255, 0)  # Green color
    elif class_id in unattended_item_classes:
            label = f"unattended {class_name} {tracker_id}" if unattended else f"{class_name} {tracker_id}"
            color = (0, 0, 255) if unattended else (0, 255, 0)
    elif class_name in staff_and_crew_classes:  # Handling for specific airline crew and airport staff
        label = f"{class_name} {tracker_id}"
        color = (0, 255, 0)  # Green color
    else:
        return frame  # Skip drawing a box and label if it does not match the required classes

    # Ensure coordinates are integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Draw the bounding box
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Calculate the position for the label
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_x = x1
    label_y = y1 - label_size[1] if y1 - label_size[1] > 10 else y1 + label_size[1]

    # Draw the label background
    frame = cv2.rectangle(frame, (label_x, label_y - label_size[1] - 2), (label_x + label_size[0], label_y + 2), color, cv2.FILLED)

    # Put the label text on the frame
    frame = cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

    return frame

# Function to log detections
def log_event(event_message):
    # Prepend the timestamp within the log_event function
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"{timestamp}: {event_message}\n"
    with open("detection_log.txt", "a") as log_file:
        log_file.write(full_message)

#Start video capture:
#cap = cv2.VideoCapture(0)#0 for default camera;
cap = cv2.VideoCapture("Recording 2024-04-12 003613.mp4")


#### Configurable email notification thresholds
person_count_threshold = 3  # for crowd detection
notification_cooldown = 300  # notification frequency limit in seconds
person_minimum_detection_duration = 5  # crowd persistence threshold in seconds
suitcase_alert_threshold = 15 # Unattended suitcase alert threshold in seconds
report_period = 0.5  # reporting period in minutes
trash_count_threshold = 5 # for trash detection
trash_minimum_detection_duration = 5 # trash persistence threshold in seconds

#initialize parameters
last_email_time = time.time() - notification_cooldown  
person_detected_start_time = None
periodic_report_time = datetime.now()
unique_tracker_ids = set()
item_tracker = {}
alerted_items = set()
trash_detection_start_time = None
trash_alert_sent = False
unique_westjet_crew_ids = set()
unique_airport_staff_ids = set()
unique_aircanada_crew_ids = set()

# Main loop to process video frames
person_boxes = []
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Reset person boxes for each frame
    person_boxes = [box for box, class_id in zip(detections.xyxy, detections.class_id) if model.names[class_id].lower() == 'person']
    # Reset person count for each frame
    frame_person_count = 0

    for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
        class_name = model.names[class_id].lower()
    # For unattended items
    for tracker_id, box, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
        if class_id in unattended_item_classes:  # Check if the class ID is in the unattended item list
            class_name = unattended_item_classes[class_id]  # Get the item name
            if not check_proximity(person_boxes, box):  # Check proximity of any person to the item
                if tracker_id not in item_tracker:  
                    item_tracker[tracker_id] = time.time()  # Start or update the timer for unattended item
                elif time.time() - item_tracker[tracker_id] > suitcase_alert_threshold:
                    frame = annotate_frame(frame, *box, tracker_id, class_id, unattended=True)  # Annotate
                    # Send email alert if not already alerted
                    if tracker_id not in alerted_items:  # Avoid repeating alert
                        send_email("Alert: Unattended Item", f"Unattended {class_name} detected.")
                        alerted_items.add(tracker_id)
                        log_event(f"Alert - Unattended {class_name} Detected")
                    continue
            else:
                # If a person is close, reset the timer and remove from the alert list
                if tracker_id in item_tracker:
                    del item_tracker[tracker_id]
                if tracker_id in alerted_items:
                    alerted_items.remove(tracker_id) 

        # Periodic report
            # Count persons
        if class_id == 0:  
            unique_tracker_ids.add(tracker_id) 
        elif class_id == 98:  #for WestJet crew
            unique_westjet_crew_ids.add(tracker_id)
        elif class_id == 99:  #for airport staff
            unique_airport_staff_ids.add(tracker_id)
        elif class_id == 100:  #for Air Canada crew
            unique_aircanada_crew_ids.add(tracker_id)    
               
            # Send a report with the count of unique people detected
        if datetime.now() >= periodic_report_time + timedelta(minutes=report_period):
            send_email("periodic Traffic Report",
                       f"In the last {report_period} minutes, {len(unique_tracker_ids)} people were detected.")
            log_event(f"Periodic Traffic Report - In the last {report_period} minutes, {len(unique_tracker_ids)} people were detected.")
            # Reset for the next period
            unique_tracker_ids.clear()
            periodic_report_time = datetime.now()

        # Person detection report
        if model.names[class_id].lower() == 'person':
            frame_person_count += 1 
        if frame_person_count > 0:
            if person_detected_start_time is None:
                person_detected_start_time = time.time()
            # Check if the accumulated count meets the threshold and if the minimum duration and cooldown have passed
            elif (frame_person_count >= person_count_threshold and 
                time.time() - person_detected_start_time >= person_minimum_detection_duration and 
                time.time() - last_email_time >= notification_cooldown):
                send_email("Alert: Crowd", f"{frame_person_count} people detected.")  # the email text
                log_event(f"Alert: High Person Count - {frame_person_count} people detected.")
                last_email_time = time.time()
                person_detected_start_time = None  # Reset after sending email
        else:
            person_detected_start_time = None
        
        # Trash detection report
        current_frame_trash_count = sum(80 <= class_id <= 97 for class_id in detections.class_id)
        if current_frame_trash_count >= trash_count_threshold:
            if trash_detection_start_time is None:
                # Start timing the detection period
                trash_detection_start_time = time.time()
            elif not trash_alert_sent and (time.time() - trash_detection_start_time > trash_minimum_detection_duration):
                # Send an email if the condition persists for more than 10 seconds
                send_email("Alert: Trash", f"{len(current_frame_trash_count)} trash detected.")
                log_event(f"Alert: {len(current_frame_trash_count)} trash detected.")
                trash_alert_sent = True  # Prevent sending multiple emails for the same event
        else:
            # Reset the timer if the condition is not met
            trash_detection_start_time = None
            trash_alert_sent = False
        
        # Annotate normally if not a suitcase or not unattended
        frame = annotate_frame(frame, *box, tracker_id, class_id)
 
    cv2.imshow("YOLOv8 Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()