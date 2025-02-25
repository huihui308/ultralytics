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
    # yolo11n-4heads.yaml: 24
    CFG = "david/detect/prim_detect/cfg/11/yolo11-4heads.yaml"

    model = YOLO(CFG)

    # model.load('runs/detect/yolo11n-4heads-class12-cbd-wangjing-yizhuang/weights/best.pt')

    model.train(
                # data='david/detect/prim_detect/datasets/uavdt.yaml',
                data='david/detect/prim_detect/datasets/visdrone.yaml',
                seed=100,
                cache=False,
                imgsz=640,
                epochs=300,
                # batch=24,
                batch=8,
                close_mosaic=10,
                device=[0,1],
                optimizer='AdamW', # SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
                # project='runs/train',
                # name='exp',
    )