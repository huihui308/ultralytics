

# Prepare datasets
delete cache files:
```
$ rm -rf /home/david/dataset/GRDDC/GRDDC2022/train.cache
$ rm -rf /home/david/dataset/GRDDC/GRDDC2022/val.cache
```

generate label directory:
```
$ cd yolov5-7.0
$ python3 data/GRDDC2022/generate_train_val_txt.py --class_file ./data/GRDDC2022/damage_classes.txt --input_dir /home/david/dataset/GRDDC/GRDDC2022

$ python3 scripts/xml2yolo.py --class_file ./data/GRDDC2022/damage_classes.txt --input_file /home/david/dataset/GRDDC/GRDDC2022/train.txt
$ python3 scripts/xml2yolo.py --class_file ./data/GRDDC2022/damage_classes.txt --input_file /home/david/dataset/GRDDC/GRDDC2022/val.txt
```

add pothole dataset:
```
$ python3 scripts/add_pothole_dataset.py --pothole_dir /home/david/dataset/GRDDC/GRDDC2022/Pothole.v1-raw.yolov5pytorch --train_file /home/david/dataset/GRDDC/GRDDC2022/train.txt --val_file /home/david/dataset/GRDDC/GRDDC2022/val.txt
```

## Create test data
```
$ cd david/datasets/
$ python3 generate_test_txt.py --input_dir=/home/david/dataset/detect/CBD/n2s_20220414_1800 --output_dir=./output
```


# Train
copy yolov8n.pt to project directory.
```
$ conda activate yolov8_3.8
$ yolo task=detect mode=train model=david/model/yolov8n.pt data=david/grddc/data/grddcDet.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=grddc

or

$ yolo cfg=default_copy.yaml

```

# Test
```
$ conda activate yolov8_3.8
$ yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
$ yolo predict model=runs/detect/train/weights/best.pt source=david/datasets/output/test/
```
There are results in './runs/detect/predict'
