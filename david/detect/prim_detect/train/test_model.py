#!/usr/bin/python3

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


if __name__ == '__main__':
    # CFG = "david/detect/prim_detect/cfg/v10/yolov10n-c3k2.yaml"
    CFG = "david/detect/prim_detect/cfg/11/yolo11t-4heads.yaml"

    SOURCE = "/home/david/dataset/class11-cbd-wangjing-yizhuang/train/images/CBD_cuiwei_SN_0058_044879538762.jpg"

    model = YOLO(CFG)
    # model_info(model)
    # print(model)
    
    model(SOURCE)