#!/usr/bin/python3
"""
    cp /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/nn /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/nn-bak

    cp -rf ultralytics/nn /home/david/anaconda3/envs/V8/lib/python3.8/site-packages/ultralytics/

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

    # model.load('runs/detect/yolo11t0-58epoches-4heads-class11-cbd-wangjing-yizhuang/weights/best.pt')

    model.train(
                data='david/detect/prim_detect/datasets/yolo_class4.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=24,
                close_mosaic=10,
                device=[0,1],
                optimizer='SGD', # using SGD
                # project='runs/train',
                # name='exp',
    )