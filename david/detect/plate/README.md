
# Prepare plate dataset
First, you should enter into the dataset directory, such as '/home/david/dataset/lpd_lpr/yolov8', and then soft link ./script/ python files to '/home/david/dataset/lpd_lpr/yolov8'.
```
$ cd /home/david/dataset/lpd_lpr/yolov8
$ ln -s /home/david/code/yolo/ultralytics/david/detect/plate/script/* ./
```

There are two types of data.
- The dataset contains vehicle and plate data which is labelme format.
- CCPD dataset

## Check data
You can execute folowing command to examine data.
```
$ python3 check_yolo_plate_data.py --dataset_dir=./ --result_dir=./output
```

## Split labelme dataset which contains vehicles and plates
The dataset contains vehicle and plate which is labelme format. So we should split vehicle which contains plate from image and calculate plate pos.

```
$ python3 labelme_split_plate_to_yolo.py --labelme_dir=/home/david/dataset/detect/echo_park --result_dir=./
$ python3 labelme_split_plate_to_yolo.py --labelme_dir=/home/david/dataset/detect/CBD --result_dir=./
```

## CCPD dataset
In this dataset, you should not split plate from whole image, so you only soft link file to current directory.

Because of there is no txt file for CCPD, so you should generate txt file first.
```
$ python3 ccpd_process.py --ccpd_dir=/home/david/dataset/lpd_lpr/detect_plate_datasets
```

In end, you can soft link plate data to the datasets.
```
$ python3 add_plate_to_yolo.py --plate_dir=/home/david/dataset/lpd_lpr/detect_plate_datasets --result_dir=./
```


# Train

Copy 'david/detect/plate/train.py' to ultralytics root directory. In ultralytics root directory.
```
$ python3 train.py
```