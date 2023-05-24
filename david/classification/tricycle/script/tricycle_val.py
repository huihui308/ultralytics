# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load an official model
model = YOLO('/home/david/code/yolo/ultralytics/runs/classify/train/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy