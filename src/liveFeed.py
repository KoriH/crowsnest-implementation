import cv2
import supervision as sv
import threading
import queue
from ultralytics import YOLO

stream_URL = "rtsp://Live:CameraDemo24!@192.168.0.252/view.html?mode=l&tcp"
model_name = "yolov8n.pt"
stop_threads = False

model = YOLO(model_name)
cap = cv2.VideoCapture(stream_URL)
frame_queue = queue.Queue(maxsize=100)

def read_frames(cap, frame_queue):
    while not stop_threads:
        success, frame = cap.read()
        if success:
            frame_queue.put(frame)
        else:
            break


thread = threading.Thread(target=read_frames, args=(cap, frame_queue))
thread.start()


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
            stop_threads = True
            break

thread.join()

cap.release()
cv2.destroyAllWindows()