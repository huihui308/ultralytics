#!/usr/bin/python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
from ultralytics import YOLO


if __name__ == '__main__':
    CFG = "ultralytics/cfg/models/11/yolo11-cls.yaml"

    model = YOLO(CFG)  # build a new model from YAML

    # model.load('yolov8n-cls.pt')

    model.train(
                data='/home/david/dataset/classification/tricycle_train_data',
                cache=False,
                imgsz=128,
                epochs=300,
                batch=4096,
                # close_mosaic=10,
                device=[0,1],
                optimizer='SGD', # using SGD
                # project='runs/train',
                # name='exp',
    )

    # # Train the model
    # #model.train(data='mnist160', epochs=100, imgsz=64)
    # model.train(data='/home/david/code/yolo/ultralytics/david/classification/tricycle/datasets/tricycle_datasets', epochs=100, imgsz=64)