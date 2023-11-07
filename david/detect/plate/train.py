import os, sys
# os.environ["OMP_NUM_THREADS"]='2'
#sys.path.append('../../../')

from ultralytics import YOLO
# Load a model
#model = YOLO('ultralytics/models/v8/yolov8-lite-t-pose.yaml')  # build a new model from YAML
model = YOLO('david/detect/plate/cfg/yolov8-lite-t-pose.yaml')  # build a new model from YAML
#model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  
#model = YOLO('david/detect/plate/weights/yolov8-lite-t-plate.pt')  # load a pretrained model (recommended for training)

# Train the model
#model.train(data='v8_plate.yaml', epochs=100, imgsz=160, batch=16, device=[0])
model.train(data='david/detect/plate/datasets/v8_plate.yaml', epochs=300, imgsz=160, batch=512, device=[0])