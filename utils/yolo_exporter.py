from ultralytics import YOLO

m = YOLO("weights/yolo11n.pt")
m.export(format="rknn", opset=19, name="rk3588", imgsz=640)