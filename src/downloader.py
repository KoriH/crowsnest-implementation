from ultralytics import YOLO
import cv2 
import supervision as sv

cap = cv2.VideoCapture("Recording 2024-04-12 003613.mp4")

model = YOLO("yolov8n.pt")

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('annotated_video.avi', fourcc, fps, (width, height))

while True:
    success, frame = cap.read()
    if success:

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [model.model.names[class_id] for class_id in detections.class_id]

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        out.write(annotated_frame)

        cv2.imshow('Annotated Frame', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        
        break


cap.release()
out.release()
cv2.destroyAllWindows()
