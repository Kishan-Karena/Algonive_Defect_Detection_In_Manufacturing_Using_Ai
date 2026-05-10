from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt") 

# Train
# Make sure data="data.yaml" points to the file in this folder
results = model.train(data="data.yaml", epochs=100, imgsz=640)