
# classes labels
class4: 'person', 'rider', 'car', 'lg'
class5: 'person', 'rider', 'tricycle', 'car', 'lg'
class6: 'person', 'ride', 'car', 'R', 'G', 'Y'
class11: 'person', 'bicycle', 'motorbike', 'tricycle', 'car', 'bus', 'truck', 'plate', 'R', 'G', 'Y'


# Prepare datasets
input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.
## class4
```
$ cd david/object_detect/datasets/
```
labelme format:
```
$ python3 labelme_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/CBD --output_dir=./output_class4
$ python3 labelme_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/echo_park --output_dir=./output_class4
```
ktti format
```
$ python3 kitti_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/kitti --output_dir=./output_class4
```
bdd format:
```
$ python3 bdd_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/bdd --output_dir=./output_class4
```

## class5
```
$ cd david/object_detect/datasets/
```
labelme format:
```
$ python3 labelme_to_yolo.py --class_num=5 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/echo_park --output_dir=/home/david/dataset/detect/yolo/class5
```
Argoverse-1.1 format:
```
$ python3 argoverse_to_yolo_class5.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/Argoverse-1.1 --output_dir=/home/david/dataset/detect/yolo/class5
```
ktti format
```
$ python3 kitti_to_yolo.py --class_num=5 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/kitti --output_dir=/home/david/dataset/detect/yolo/class5
```
bdd format:
```
$ python3 bdd_to_yolo.py --class_num=5 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/bdd --output_dir=/home/david/dataset/detect/yolo/class5
```
UA-DETRAC format:
```
$ python3 UA-DETRAC_to_yolo_class5.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/UA-DETRAC --output_dir=/home/david/dataset/detect/yolo/class5
```


## class6
```
$ cd david/object_detect/datasets/
$ python3 labelme_to_yolov_class6.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/CBD --output_dir=./output_class6

$ python3 labelme_to_yolov_class6.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/echo_park --output_dir=./output_class6
```

## class11
input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.
```
$ cd david/object_detect/datasets/
$ python3 labelme_to_yolov_class11.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/CBD --output_dir=./output_class11

$ python3 labelme_to_yolov_class11.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/echo_park --output_dir=./output_class11

$ python3 kitti_to_yolov_class11.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/kitti --output_dir=./output_class11
```

## Create test data
```
$ cd david/object_detect/datasets/
$ python3 generate_test_txt.py --input_dir=/home/david/dataset/detect/CBD/n2s_20220414_1800 --output_dir=./output_class4
```


# Train
copy yolov8n.pt to project directory.
```
$ cp runs/detect/obj_det_class11/weights/best.pt yolov8n.pt
```

# class4
```
$ yolo task=detect mode=train model=david/object_detect/model/cuiwei.pt data=david/object_detect/data/det_data_class4.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=obj_det_class4
```

# class5
Signal gpu:
```
$ yolo task=detect mode=train model=david/object_detect/model/yolov8n_obj_det_class4_ktti_bdd_cbd_echopark.pt data=david/object_detect/data/det_data_class5.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=echopark_class5
```
Multiple gpu:
```
$ yolo task=detect mode=train model=david/object_detect/model/yolov8n_obj_det_class4_ktti_bdd_cbd_echopark.pt data=david/object_detect/data/det_data_class5.yaml epochs=300 batch=128 device=0,1 workers=56 resume=False name=all_class5
```

# class6
```
$ yolo task=detect mode=train model=david/object_detect/model/cuiwei.pt data=david/object_detect/data/det_data_class6.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=obj_det_class6
```

# class11
```
$ yolo task=detect mode=train model=david/object_detect/model/cuiwei.pt data=david/object_detect/data/det_data_class11.yaml epochs=300 batch=32 device=0 workers=56 resume=False name=obj_det_class11
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



