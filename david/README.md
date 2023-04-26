

# Prepare datasets
## labelme_to_yolov8.py
input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.
```
$ cd david/datasets/
$ python3 labelme_to_yolov8.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output
```

## Create test data
```
$ cd david/datasets/
$ python3 generate_test_txt.py --input_dir=/home/david/dataset/detect/CBD/n2s_20220414_1800 --output_dir=./output
```


# Train
```
$ yolo task=detect mode=train model=david/model/yolov8n.pt data=david/config/primaryDet.yaml epochs=100 batch=32 device=0 workers=56 resume=False
```



