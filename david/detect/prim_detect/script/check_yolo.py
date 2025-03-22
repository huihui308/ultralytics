#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os


def show_convert_results(image_path, label_path):
    for file_name in os.listdir(image_path):
        # print(file_name)
        file_prefix = file_name.split('.')[0]
        file_path_name = os.path.join(image_path, file_name)
        label_path_name = os.path.join(label_path, file_prefix + '.txt')
        # print(label_path_name, file_path_name)
        if not os.path.exists(label_path_name):
            print("{} not exists".format(label_path_name))
            continue
        # 读取YOLO格式的标签文件
        image = cv2.imread(file_path_name)
        img_h, img_w, _ = image.shape
        with open(label_path_name, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                object_category, x_center, y_center, width, height = data
                # 计算边界框坐标
                x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
                x_center *= img_w
                y_center *= img_h
                width *= img_w
                height *= img_h
                x1, y1, x2, y2 = int(x_center - width/2), int(y_center - height/2), int(x_center + width/2), int(y_center + height/2)
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, object_category, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.imwrite('./test.jpg', image)
        cv2.imshow('UAVDT', image)
        cv2.waitKey(0)
    return


if __name__ == "__main__":
    image_path = r'/home/david/dataset/drone/uavdt-split/val/images'
    label_path = r'/home/david/dataset/drone/uavdt-split/val/labels'
    #image_path = r'/home/david/dataset/drone/VisDroneOrigin/VisDrone2019-DET-train/images'
    #label_path = r'/home/david/dataset/drone/VisDroneOrigin/VisDrone2019-DET-train/labels'
    
    show_convert_results(image_path, label_path)
