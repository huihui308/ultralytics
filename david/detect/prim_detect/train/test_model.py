#!/usr/bin/python3

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


if __name__ == '__main__':
    CFG = "david/detect/plate/cfg/yolov8-lite-t-pose.yaml"
    # CFG = "david/detect/prim_detect/cfg/11/yolo11-4heads.yaml"

    SOURCE = "/home/david/dataset/class11-cbd-wangjing-yizhuang/train/images/CBD_cuiwei_SN_0058_044879538762.jpg"

    model = YOLO(CFG)

    # model_info(model)
    # print(model)
    
    model(SOURCE)