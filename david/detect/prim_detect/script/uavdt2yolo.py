#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    $ python3 uavdt2yolo.py

    https://github.com/forever208/yolov5_train_on_UAVDT
"""
from tqdm import tqdm
import os, shutil, random


def organise_image_folders(old_dir, output_dir)->None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # folder_names = os.listdir(old_dir)
    # for folder in folder_names:
    for folder in tqdm(os.listdir(old_dir), ncols=100):
        # folder_path = old_dir + '/' + folder    # '../../UAV-benchmark-M/M0403'
        folder_path = os.path.join(old_dir, folder, 'img1')
        img_filename_ls = os.listdir(folder_path)    # 'img000061.jpg'
        for img_filename in img_filename_ls:
            # '../../UAV-benchmark-M/M0403/img000061.jpg'
            # old_img_path = old_dir + '/' + folder + '/' + img_filename
            old_img_path = os.path.join(old_dir, folder, 'img1', img_filename)
            # ../../dataset/images/all/M0403_000061.jpg
            # output_img_path = output_dir + '/' + folder + '_' + img_filename[-10:]
            output_img_path = os.path.join(output_dir, folder + '_' + img_filename[-10:])
            # copy images from old path tp new path
            # shutil.copyfile(old_img_path, output_img_path)
            os.symlink(old_img_path, output_img_path)
        # print('image folder copy finished: ', folder)
    print('all images has been copied into: ', output_dir)
    return


def organise_txt_labels(old_dir, output_dir)->None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    IMG_W = 1024
    IMG_H = 540

    # video_label_txts = os.listdir(old_dir)
    # for each_txt in video_label_txts:    # 'M1006_gt_whole.txt'
    for each_txt in tqdm(os.listdir(old_dir), ncols=100):
        # if each_txt[-6:] == 'gt.txt':
        if each_txt[-13:] == '_gt_whole.txt':
            video_name = each_txt[:5]    # 'M1006'
            txt_path = os.path.join(old_dir, each_txt)  # '../../UAV-benchmark-MOTD_v1.0/GT/M1006_gt_whole.txt'
            # read txt
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_ls = line.split(',')    # ['1089', '20', '461', '207', '21', '31', '1', '1']
                    img_num = line_ls[0]    # '1089'
                    img_six_num = (6-len(img_num))*'0' + str(img_num)    # '001089'

                    # transform [x1, y1, w, h] to [xc, yc, w, h]
                    org_xc = int(line_ls[2]) + int(line_ls[4])/2
                    org_yc = int(line_ls[3]) + int(line_ls[5])/2
                    org_w = int(line_ls[4])
                    org_h = int(line_ls[5])

                    # print(each_txt, line, line_ls[6], line_ls[7], line_ls[8])
                    ann_str = str(line_ls[8]).strip()
                    # print(type(line_ls[8]), line_ls[8], ann_str)
                    # type_str = "3"
                    if ann_str == "1":
                        type_str = "0"            # car
                    elif ann_str == "2":
                        type_str = "1"            # truck
                    elif ann_str == "3":
                        type_str = "2"            # bus
                    # print(type_str)

                    # remove wrong labels (there are some wrongly annotated bbox, most of them are very big)
                    if org_w > (IMG_W / 6) and org_h > (IMG_H / 6):
                        continue

                    # convert coordinates scale from image size to [0, 1]
                    xc = float(org_xc / IMG_W)
                    yc = float(org_yc / IMG_H)
                    w = float(org_w / IMG_W)
                    h = float(org_h / IMG_H)

                    if xc > 1:
                        print('oh no!!! xc ', xc)
                    if yc > 1:
                        print('oh no!!! yc ', yc)
                    if w > 1:
                        print('oh no!!! w ', w)
                    if h > 1:
                        print('oh no!!! h ', h)

                    # write bbox into new txt (one image corresponds to one txt label)
                    # new_txt_path = output_dir + '/' + video_name + '_' + img_six_num + '.txt'
                    new_txt_path = os.path.join(output_dir, video_name + '_' + img_six_num + '.txt')
                    with open(new_txt_path, 'a') as wr:
                        bbox = type_str + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n'
                        wr.writelines(bbox)
            # print(each_txt, ' has been parsed')
    print('all txt labels have been saved in: ', output_dir)
    return


def split_train_val(images_dir, labels_dir, tra_img_dir, val_img_dir, tra_labels_dir, val_labels_dir)->None:
    if not os.path.exists(tra_img_dir):
        os.makedirs(tra_img_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(tra_labels_dir):
        os.makedirs(tra_labels_dir)
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)

    labels_filename = os.listdir(labels_dir)    # ['M1306_000215.txt', 'M0501_000291.txt', ......]
    random.shuffle(labels_filename)    # shuffle the dataset

    tra_filename = labels_filename[:35000]
    val_filename = labels_filename[35000:]
    print('number of training images is: ', len(tra_filename))
    print('number of validation images is: ', len(val_filename))

    # training dataset
    for tra_file in tra_filename:    # 'M1306_000215.txt'
        prefix = tra_file[:12]    # 'M1306_000215'
        old_tra_img = images_dir + '/' + prefix + '.jpg'
        new_tra_img = tra_img_dir + '/' + prefix + '.jpg'
        # shutil.copyfile(old_tra_img, new_tra_img)
        os.symlink(old_tra_img, new_tra_img)

        old_tra_label = labels_dir + '/' + prefix + '.txt'
        new_tra_label = tra_labels_dir + '/' + prefix + '.txt'
        # shutil.copyfile(old_tra_label, new_tra_label)
        os.symlink(old_tra_label, new_tra_label)
    print('training images have been saved in folder: ', tra_img_dir)
    print('training labels have been saved in folder: ', tra_labels_dir)

    # validation dataset
    for val_file in val_filename:    # 'M1306_000215.txt'
        prefix = val_file[:12]    # 'M1306_000215'
        old_val_img = images_dir + '/' + prefix + '.jpg'
        new_val_img = val_img_dir + '/' + prefix + '.jpg'
        # shutil.copyfile(old_val_img, new_val_img)
        os.symlink(old_val_img, new_val_img)

        old_val_label = labels_dir + '/' + prefix + '.txt'
        new_val_label = val_labels_dir + '/' + prefix + '.txt'
        # shutil.copyfile(old_val_label, new_val_label)
        os.symlink(old_val_label, new_val_label)
    print('validation images have been saved in folder: ', val_img_dir)
    print('validation labels have been saved in folder: ', val_labels_dir)

    # remove old dataset folders
    # shutil.rmtree(images_dir)
    # shutil.rmtree(labels_dir)
    return


if __name__ == "__main__":
    old_dir = '/home/david/dataset/drone/UAVDTOrigin/UAV-benchmark-M'
    output_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/images/all'
    organise_image_folders(old_dir, output_dir)

    old_dir = '/home/david/dataset/drone/UAVDTOrigin/UAV-benchmark-MOTD_v1.0/GT'
    output_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/labels/all'
    organise_txt_labels(old_dir, output_dir)

    images_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/images/all'
    labels_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/labels/all'
    tra_img_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/train/images'
    val_img_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/val/images'
    tra_labels_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/train/labels'
    val_labels_dir = '/home/david/dataset/drone/UAVDTOrigin/yolo/val/labels'
    split_train_val(images_dir, labels_dir, tra_img_dir, val_img_dir, tra_labels_dir, val_labels_dir)




