#!/usr/bin/python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    #model = YOLO('yolov8n-cls.pt')  # load an official model
    model = YOLO('runs/classify/tricycle_20241121/weights/best.pt')  # load a custom model

    # Predict with the model
    #results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    results = model('/home/david/dataset/classification/tricycle_train_data/val/class0000/test_3_3_22109_20230723_113112_589658308459.jpg')
    # print(results)