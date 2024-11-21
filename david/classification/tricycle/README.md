
# Prepare dataset
Put 'tricycle_20230803_44300.zip' and '千方三轮车-44300帧-0825.rar' in the same directory. Such as in the directory '/home/david/dataset/classification/tricycle_raw_data'.<br/>
**Note: The label files and image files are in the same directory directly.**

<br>Process dataset:
```
$ cd david/classification/tricycle/script/
$ python3 convert_dataset.py --input_dir=/home/david/dataset/classification/tricycle_raw_data --output_dir=/home/david/dataset/classification/tricycle_train_data
```


# Train
```
$ conda activate V8
$ ln -s david/classification/tricycle/script/tricycle_train.py ./
$ python3 tricycle_train.py
or
$ yolo task=classify mode=train model=yolov8n-cls.pt data={dataset.location} epochs=50 imgsz=128
```


# Val
```
$ conda activate V8
$ yolo task=classify mode=val model={HOME}/runs/classify/train/weights/best.pt data={dataset.location}
```


# Inference
```
$ conda activate V8
$ ln -s david/classification/tricycle/script/tricycle_predict.py ./
$ python3 tricycle_predict.py
or
$ yolo task=classify mode=predict model={HOME}/runs/classify/train/weights/best.pt conf=0.25 source={dataset.location}/test/overripe
```


# Export
```
$ conda activate V8
$ ln -s david/classification/tricycle/script/tricycle_export.py ./
$ python3 tricycle_export.py
```

# Reference

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb#scrollTo=Wjc1ctZykYuf


```
前类型："未知", "有棚", "无棚"
后类别："未知", "有棚", "无棚", "有箱"
颜 色："未知", "黑色", "白色", "灰色", "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "棕色", "粉色"
用 途："未知", "载人", "货运", "快递", "外卖"
厂 家："未知", "顺丰", "申通", "圆通", "中通", "邮政", "京东", "德邦", "韵达", "百世", "苏宁", "天猫", "极兔", "海皇", "嘉德", "品骏", "丹鸟", "多点", "博信达", "宅急送"
```

