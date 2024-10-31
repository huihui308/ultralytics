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
    # yolo11s-4heads
    CFG = "david/detect/prim_detect/cfg/11/yolo11-4heads.yaml"
    # CFG = "david/detect/prim_detect/cfg/v10/yolov10n-c3k2.yaml"
    
    model = YOLO(CFG)

    # model.load('yolo11s.pt') # loading pretrain weights

    model.train(data='david/detect/prim_detect/datasets/yolov10_class11.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                device=[0,1],
                optimizer='SGD', # using SGD
                # project='runs/train',
                # name='exp',
    )