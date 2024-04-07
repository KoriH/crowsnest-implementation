from ultralytics import YOLO
import supervision as sv
import cv2
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("rtsp://service:Password!234@192.168.2.3/view.html?mode=l&tcp")

annotate = sv.BoxAnnotator()

classNames = ["person", "backpack", "umbrella",
              "handbag", "suitcase", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "laptop", "cell phone",
              "teddy bear"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()