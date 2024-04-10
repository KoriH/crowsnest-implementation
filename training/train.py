from ultralytics import YOLO
import os 

model = YOLO("yolov8n.pt")

results = model.train(
    data="config.yaml",
    patience=25,
    imgsz=640,
    device=0,
    pretrained=True,
    optimizer='SGD',
    epochs=40)

metrics = model.val()