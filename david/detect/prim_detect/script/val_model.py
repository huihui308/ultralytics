#!/usr/bin/python3
"""
$ nohup python3 david/detect/prim_detect/script/val_model.py > log.txt 2>&1 &

"""
from ultralytics import YOLO
import os
from pathlib import Path

# Step 1: Load the YOLOv11 model
# model = YOLO("/home/david/code/ultralytics/best.pt")  # Replace with your trained model path if needed
model = YOLO("/home/david/code/ultralytics/runs/detect/train/weights/best.pt")  # Replace with your trained model path if needed

# Step 2: Define paths to the validation directory
validation_dir = Path("/home/david/dataset/drone/UAVDTOrigin/consumer/UAVDT/val")  # Update this path
#validation_dir = Path("/home/david/dataset/drone/VisDroneOrigin/VisDrone2019-DET-val")  # Update this path

images_dir = validation_dir / "images"
labels_dir = validation_dir / "labels"

# Step 3: Validate the dataset and calculate mAP50
results = model.val(
    data="/home/david/code/ultralytics/david/detect/prim_detect/datasets/uavdt.yaml",  # Path to your dataset YAML file (explained below)
    #data="/home/david/code/ultralytics/david/detect/prim_detect/datasets/visdrone.yaml",  # Path to your dataset YAML file (explained below)
    imgsz=640,         # Input image size
    batch=16,          # Batch size
    conf=0.25,         # Confidence threshold
    iou=0.5,           # IoU threshold for mAP50
    device=[0,1],
    # device="cpu",      # Use "cuda" if GPU is available
    save_json=True,   # Set to True if you want to save results as JSON
    plots=True         # Generate plots for analysis
)

# Step 4: Access mAP50 from the results
metrics = results.results_dict
map50 = metrics['metrics/mAP50(B)']  # mAP50 for bounding box detection
print(metrics)

print("\nResults:")
print("-------------------------")

print(f"mAP: {results.box.map}")
print(f"mAP50: {results.box.map50}")
# print(f"mAP50: {map50:.4f}")
print(f"mAP75: {results.box.map75}")

# print(results.speed['inference'], float(results.speed['inference']))
fps = int(1000/float(results.speed['inference']))
print(f"FPS: {fps}")
print("-------------------------")
