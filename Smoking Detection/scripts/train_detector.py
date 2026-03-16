from ultralytics import YOLO
import os
os.makedirs("models", exist_ok=True)

# Use YOLOv8 nano pretrained
model = YOLO("yolov8n.pt")

# Train detector
model.train(
    data="datasets/detector/smoking_detector_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="smoking_detector"
)

# Save
model.save("models/detector_model.pt")
print("Detector model saved!")
