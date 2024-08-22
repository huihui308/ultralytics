# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.pt')  # load an official model
model = YOLO('/home/david/code/yolo/ultralytics/runs/classify/train/weights/best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')