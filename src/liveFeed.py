import cv2
import supervision as sv
import threading
import queue
from ultralytics import YOLO

frame_queue = queue.Queue(maxsize=100)

def read_frames(cap, frame_queue):
    while True:
        success, frame = cap.read()
        if success:
            frame_queue.put(frame)
        else:
            break

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("rtsp://service:Password!234@192.168.1.123/view.html?mode=l&tcp")

threading.Thread(target=read_frames, args=(cap, frame_queue)).start()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
            model.model.names[class_id]
            for class_id
            in detections.class_id
        ]

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow('Webcam', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()