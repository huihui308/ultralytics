#!/usr/bin/python3
"""
    $ cp /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/nn /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/nn-bak

    $ cp -rf ultralytics/nn /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/
"""
# import sys
# sys.path.insert(0, '/home/david/code/ultralytics')

# import warnings
# warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # CFG = "david/detect/plate/cfg/yolo11-pose.yaml"
    CFG = "david/detect/plate/cfg/yolov8-lite-t-pose.yaml"

    model = YOLO(CFG)

    # model.load('runs/detect/yolo11t0-58epoches-4heads-class11-cbd-wangjing-yizhuang/weights/best.pt')

    # imgsz=320: 576    imgsz=320: 960
    model.train(
                data='david/detect/plate/datasets/v8_plate.yaml',
                # data='david/detect/prim_detect/datasets/yolov10_class11.yaml',
                cache=False,
                imgsz=320,
                epochs=300,
                batch=576,
                close_mosaic=10,
                device=[0,1],
                optimizer='SGD', # using SGD
                # project='runs/train',
                # name='exp',
    )