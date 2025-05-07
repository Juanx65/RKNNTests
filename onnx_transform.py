from ultralytics import YOLO

# Load the model
model = YOLO("weights/yolov8n.pt")

# Export the model to ONNX format
model.export(format="onnx",imgsz=640,batch=1, opset=12)
