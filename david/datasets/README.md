





```
$ python3 rename.py ./WE/
$ python3 rename.py ./SN/
$ labelme2yolo --json_dir ./WE/ --val_size 0.15 --test_size 0.15
$ labelme2yolo --json_dir ./SN/ --val_size 0.15 --test_size 0.15
```



# cmd
```
python3 primary_detect_prepare_data.py --target_width=1248 --target_height=384 --input_dir=./../../dataset/CBD --output_dir=./../data
```

input_dir: Dataset which generate by labelme, it must contains a label file and a jpg file.





# labelme_to_yolov8.py
python3 labelme_to_yolov8.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output

