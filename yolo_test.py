from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict("bus.jpg",save=True)

