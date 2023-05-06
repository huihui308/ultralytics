
# Prepare datasets
## type6.py
input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.
```
$ cd david/object_detect/datasets/
$ python3 labelme_to_yolov8_type6.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output_type6

$ python3 labelme_to_yolov8_type6.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/ecopark/haitian-20230428 --output_dir=./output_type6
```

## type11.py
input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.
```
$ cd david/object_detect/datasets/
$ python3 labelme_to_yolov8_type11.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output_type11

$ python3 labelme_to_yolov8_type11.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/ecopark/haitian-20230428 --output_dir=./output_type11
```

## Create test data
```
$ cd david/object_detect/datasets/
$ python3 generate_test_txt.py --input_dir=/home/david/dataset/detect/CBD/n2s_20220414_1800 --output_dir=./output
```


# Train
copy yolov8n.pt to project directory.
```
$ yolo task=detect mode=train model=david/object_detect/model/yolov8n.pt data=david/object_detect/data/det_data_type11.yaml epochs=300 batch=32 device=0 workers=56 resume=False

$ yolo task=detect mode=train model=david/object_detect/model/cuiwei.pt data=david/object_detect/data/det_data_type11.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=obj_det
```

# Test
```
$ yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
$ yolo predict model=runs/detect/train/weights/best.pt source=david/datasets/output/test/
```
There are results in './runs/detect/predict'


# Deploy

## Deploy on deepstream-6.0
Reference: https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLOv8.md

```
$ cd david/object_detect/deploy
```
Copy weight file to current directory.
$ cp ./../../../runs/detect/obj_det/weights/best.pt ./
```
Generate the cfg, wts and labels.txt (if available) files
```
$ python3 gen_wts_yoloV8.py --size 640 -w best.pt
```
Copy the generated cfg, wts and labels.txt (if generated), files to the deepstream folder.



