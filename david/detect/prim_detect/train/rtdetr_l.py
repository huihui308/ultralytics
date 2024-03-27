from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
#model = RTDETR('rtdetr-l.pt')
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-n.yaml')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data='david/detect/prim_detect/datasets/rtdetr_l.yaml', epochs=300, imgsz=640)
results = model.train(data='david/detect/prim_detect/datasets/det_data_class5.yaml', epochs=300, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
#results = model('path/to/bus.jpg')