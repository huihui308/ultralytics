#!/usr/bin/python3

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    # model = YOLO("ultralytics/cfg/models/11/yolo11-4heads.yaml")
    # model = YOLO('ultralytics/cfg/models/11/yolo11-EMA_attention.yaml')

    model.load('yolo11n.pt') # loading pretrain weights

    model.train(data='david/detect/prim_detect/datasets/yolov10_class11.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=48,
                close_mosaic=10,
                device=[0,1],
                optimizer='SGD', # using SGD
                # project='runs/train',
                # name='exp',
    )