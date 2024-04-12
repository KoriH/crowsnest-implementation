from ultralytics import YOLO
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

torch.cuda.is_available()

model = YOLO("yolov8n.pt")

results = model.train(
    data="config.yaml",
    device=0,
    optimizer='SGD',
    epochs=100)

metrics = model.val()