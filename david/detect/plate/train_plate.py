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
    # yolo11to-4heads
    CFG = "ultralytics/models/v8/yolov8-lite-t-pose.yaml"
    # CFG = "david/detect/prim_detect/cfg/v10/yolov10n-c3k2.yaml"
    
    model = YOLO(CFG)

    # model.load('runs/detect/yolo11t0-58epoches-4heads-class11-cbd-wangjing-yizhuang/weights/best.pt')

    model.train(
                data='david/detect/prim_detect/datasets/yolo_class12.yaml',
                # data='david/detect/prim_detect/datasets/yolov10_class11.yaml',
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



# # Load a model
# #model = YOLO('ultralytics/models/v8/yolov8-lite-t-pose.yaml')  # build a new model from YAML
# model = YOLO('david/detect/plate/cfg/yolov8-lite-t-pose.yaml')  # build a new model from YAML
# #model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  
# #model = YOLO('david/detect/plate/weights/yolov8-lite-t-plate.pt')  # load a pretrained model (recommended for training)

# # Train the model
# #model.train(data='v8_plate.yaml', epochs=100, imgsz=160, batch=16, device=[0])
# model.train(data='david/detect/plate/datasets/v8_plate.yaml', epochs=300, imgsz=160, batch=512, device=[0])