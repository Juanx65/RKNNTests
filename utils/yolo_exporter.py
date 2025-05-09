from ultralytics import YOLO

m = YOLO("weights/yolo11n.pt")
m.export(format="rknn", opset=19, name="rk3588", imgsz=640)

#from ultralytics import YOLO

# Load the model
#model = YOLO("weights/yolov8m.pt")

# Export the model to ONNX format
#model.export(format="onnx",imgsz=640,batch=1, opset=12)