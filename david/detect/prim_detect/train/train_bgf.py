import os, sys
# os.environ["OMP_NUM_THREADS"]='2'
sys.path.insert(0, '/home/david/code/yolo/ultralytics')
#print(sys.path)

from ultralytics import YOLO
# Load a model
model = YOLO(model='david/detect/prim_detect/architecture/bfg-yolov8n.yaml')  # build a new model from YAML
#model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  
#model = YOLO('david/detect/prim_detect/model/bgf_yolo.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(cfg='david/detect/prim_detect/cfg/bfg_tiny.yaml', data='david/detect/prim_detect/datasets/primary7.yaml')
#model.train(data='david/detect/prim_detect/datasets/det_data_class5.yaml', epochs=300, imgsz=640, batch=16, device=[0])