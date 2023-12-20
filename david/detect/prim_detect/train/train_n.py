import os, sys
# os.environ["OMP_NUM_THREADS"]='2'
#sys.path.append('../../../')

from ultralytics import YOLO
# Load a model
model = YOLO('david/detect/prim_detect/cfg/yolov8s.yaml')  # build a new model from YAML
#model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  
#model = YOLO('david/detect/plate/weights/yolov8-lite-t-plate.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='david/detect/prim_detect/datasets/det_data_class5.yaml', epochs=300, imgsz=640, batch=16, device=[0])