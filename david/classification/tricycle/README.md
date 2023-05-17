






# Train
```
$ yolo task=classify mode=train model=yolov8n-cls.pt data={dataset.location} epochs=50 imgsz=128
```

# Val
```
$ yolo task=classify mode=val model={HOME}/runs/classify/train/weights/best.pt data={dataset.location}
```

# Inference
```
$ yolo task=classify mode=predict model={HOME}/runs/classify/train/weights/best.pt conf=0.25 source={dataset.location}/test/overripe
```



# Reference

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb#scrollTo=Wjc1ctZykYuf




forward: unknown, shed, open
backward: unknown, shed, open, box
Color: unknown、red、orange、yellow、green、blue、purple、pink、black、white、gray、brown
Purpose: unknown, express, takeaway, freight, passenger
brand: unknown, shunfeng, jingdong, youzheng, zhongtong, yuantong, shentong, debang, yunda, tiantian, baishi, danniao, tianmao, haihuang
前类型：未知、有棚、无棚
后类别：未知、有棚、无棚、有箱
颜色：未知、红色、橙色、黄色、绿色、蓝色、紫色、粉色、黑色、白色、灰色、棕色
用途：未知、快递、外卖、货运、载人
厂家：未知、顺丰、京东、邮政、中通、圆通、申通、德邦、韵达、天天、百世、丹鸟、天猫、海皇




[
    "前有棚", "后有棚", "后有箱", 
    "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "粉色", "黑色", "白色", "灰色", "棕色", 
    "快递", "外卖", "货运", "载人", 
    "顺丰", "京东", "邮政", "中通", "圆通", "申通", "德邦", "韵达", "天天", "百世", "丹鸟", "天猫", "海皇"
]











