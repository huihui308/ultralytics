################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2019-2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# python3 labelme_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/echo_park --output_dir=/home/david/dataset/detect/yolo/all_class4
# 
# python3 labelme_to_yolo.py --input_dir=/home/david/dataset/detect/CBD --show_statistic=True
#
################################################################################

""" Script to prepare resized images/labels for primary detect. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import json
import threading
from tqdm import tqdm
from PIL import Image
from typing import List
import os, sys, math, shutil, random, datetime, signal, argparse


categories4_list = ['person', 'rider', 'tricycle', 'car']
categories5_list = ['person', 'rider', 'tricycle', 'car', 'lg']
TQDM_BAR_FORMAT = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}'


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def term_sig_handler(signum, frame) -> None:
    sys.stdout.write('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.write('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.write('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Prepare resized images/labels dataset for LPD')
    parser.add_argument(
        "--input_dir",
        type = str,
        required = True,
        help = "Input directory to OpenALPR's benchmark end2end us license plates."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = False,
        help = "Ouput directory to resized images/labels."
    )
    parser.add_argument(
        "--class_num",
        type = int,
        required = False,
        help = "Class num. 4:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3'}, 5:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3', 'lg':'4'}, 6:{'person':'0', 'rider':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}, 11:{'person':'0', 'bicycle':'1', 'motorbike':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}"
    )
    parser.add_argument(
        "--target_width",
        type = int,
        required = False,
        help = "Target width for resized images/labels."
    )
    parser.add_argument(
        "--target_height",
        type = int,
        required = False,
        help = "Target height for resized images/labels."
    )
    parser.add_argument(
        "--show_statistic",
        type = bool,
        required = False,
        help = "Statistic labels count."
    )
    parser.add_argument(
        "--loop_cnt",
        type = int,
        required = False,
        default = 1,
        help = "Directory loop count."
    )
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lopDir0 in ("train", "val"):
        firstDir = os.path.join(output_dir, lopDir0)
        if not os.path.exists(firstDir):
            #shutil.rmtree(firstDir)
            os.makedirs(firstDir)
        for lopDir1 in ("images", "labels"):
            secondDir = os.path.join(firstDir, lopDir1)
            if not os.path.exists(secondDir):
                os.makedirs(secondDir)
    return


class DealDirFilesThread(threading.Thread):
    def __init__(self, deal_dir:str, output_dir:str, output_size:List[int]):
        threading.Thread.__init__(self)
        self.deal_dir = deal_dir
        self.output_dir = output_dir
        self.output_size = output_size

    def run(self):
        sys.stdout.write('\r>> {}: Deal dir: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.deal_dir))
        sys.stdout.flush()
        img_list = []
        label_list = []
        for root, dirs, files in os.walk(self.deal_dir):
            #print(len(files))
            for file in sorted(files):
                #print(os.path.splitext(file)[-1])
                if os.path.splitext(file)[-1] == '.json':
                    label_list.append( os.path.join(root, file) )
                else:
                    img_list.append( os.path.join(root, file) )
        if len(label_list) != len(img_list):
            sys.stdout.write('\r>> {}: File len {}:{} err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(label_list), len(img_list)))
            sys.stdout.flush()
            os._exit(2)
        #print(img_list)
        img_label_list = []
        for i in range( len(img_list) ):
            img_label = []
            img_label.append(img_list[i])
            img_label.append(label_list[i])
            img_label_list.append(img_label[:])
        random.shuffle(img_label_list)
        #print(img_label_list)
        for (i, img_label) in enumerate(img_label_list):
            img_file = img_label[0]
            label_file = img_label[1]
            if os.path.splitext(img_file)[0] != os.path.splitext(label_file)[0]:
                sys.stdout.write('\r>> {}: Image file {} and label file {} not fit err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file, label_file))
                sys.stdout.flush()
                os._exit(2)
            deal_one_image_label_files(img_file, label_file, self.output_dir, i, self.output_size)
        return


# https://blog.csdn.net/Just_do_myself/article/details/118656543
# 封装resize函数
def resize_img_keep_ratio(img_name,target_size):
    img = cv2.imread(img_name) # 读取图片
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new


def GenerateKITTIDataset(img_file:str, label_dict_list, output_dir:str, deal_cnt:int, output_size:List[int])->None:
    """ Create KITTI dataset. """
    sys.stdout.write('\r>> {}: Deal file {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file))
    sys.stdout.flush()

    save_image_dir = None
    save_label_dir = None
    if ( ((deal_cnt % 10) == 8) or ((deal_cnt % 10) == 9) ):
        save_dir = os.path.join(output_dir, "testing")
        save_image_dir = os.path.join(save_dir, "images")
        save_label_dir = os.path.join(save_dir, "labels")
    else:
        save_dir = os.path.join(output_dir, "training")
        save_image_dir = os.path.join(save_dir, "images")
        save_label_dir = os.path.join(save_dir, "labels")

    dir_name, full_file_name = os.path.split(img_file)
    sub_dir_name = dir_name.split('/')[-1]
    save_file_name = sub_dir_name + "_" + str(random.randint(0, 99999999)).zfill(8)
    #print( save_file_name )
    
    # resize labels
    w, h = output_size
    img = cv2.imread(img_file)
    (height, width, _) = img.shape
    ratio_w = float( float(w)/float(width) )
    ratio_h = float( float(h)/float(height) )
    # resize images
    image = Image.open(img_file)
    tmp_image = image.resize((2560, 1440), Image.ANTIALIAS)
    scale_image = tmp_image.resize(output_size, Image.ANTIALIAS)
    scale_image.save(os.path.join(save_image_dir, save_file_name + ".jpg"))
    #shutil.copyfile(img_file, os.path.join(save_image_dir, save_file_name + ".jpg"))
    with open(os.path.join(save_label_dir, save_file_name + ".txt"), "w") as f:
        for obj_dict in label_dict_list:
            for label_str, point_list in obj_dict.items():
                #print(label_str)
                if len(point_list) == 0:
                    continue
                for one_point_list in point_list:
                    #print(one_point_list)
                    if len(one_point_list) != 2:
                        sys.stdout.write('\r>> {}: Label file point len err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        sys.stdout.flush()
                        os._exit(2)
                    x1 = float(one_point_list[0][0]) * ratio_w
                    y1 = float(one_point_list[0][1]) * ratio_h
                    x2 = float(one_point_list[1][0]) * ratio_w
                    y2 = float(one_point_list[1][1]) * ratio_h
                    f.write("{} 0.0 0 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(label_str, x1, y1, x2, y2))
    return


def labelme2_class4_yolo_data(fp, shape_obj, obj_cnt_list, img_width, img_height)->None:
    """ Create Yolo dataset. """
    type_str = None
    if shape_obj['label'] in ('person'):
        type_str = '0'
        obj_cnt_list[0] += 1
    elif shape_obj['label'] in ('bicycle', 'motorbike'):
        type_str = '1'
        obj_cnt_list[1] += 1
    elif shape_obj['label'] in ('tricycle'):
        type_str = '2'
        obj_cnt_list[2] += 1
    elif shape_obj['label'] in ('car', 'bus', 'truck'):
        type_str = '3'
        obj_cnt_list[3] += 1
    elif shape_obj['label'] in ('plate', 'plate+', 'R', 'G', 'Y', 'B'):
        return
    else:
        prRed('Label {} not support, return'.format(shape_obj['label']))
        return
    one_point_list = shape_obj['points']
    obj_width = float(one_point_list[1][0]) - float(one_point_list[0][0])
    obj_height = float(one_point_list[1][1]) - float(one_point_list[0][1])
    x_center = (float(one_point_list[0][0]) + obj_width/2)/img_width
    y_center = (float(one_point_list[0][1]) + obj_height/2)/img_height
    yolo_width = obj_width/img_width
    yolo_height = obj_height/img_height
    if (x_center <= 0.0) or (y_center <= 0.0) or (yolo_width <= 0.0) or (yolo_height <= 0.0):
        prRed('Yolo pos {} {} {} {} err, return'.format(x_center, y_center, yolo_width, yolo_height))
        return
    fp.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(type_str, x_center, y_center, yolo_width, yolo_height))
    return


def labelme2_class5_yolo_data(fp, shape_obj, obj_cnt_list, img_width, img_height)->None:
    """ Create Yolo dataset. """
    type_str = None
    if shape_obj['label'] in ('person'):
        type_str = '0'
        obj_cnt_list[0] += 1
    elif shape_obj['label'] in ('bicycle', 'motorbike'):
        type_str = '1'
        obj_cnt_list[1] += 1
    elif shape_obj['label'] in ('tricycle'):
        type_str = '2'
        obj_cnt_list[2] += 1
    elif shape_obj['label'] in ('car', 'bus', 'truck'):
        type_str = '3'
        obj_cnt_list[3] += 1
    elif shape_obj['label'] in ('R', 'G', 'Y', 'B'):
        type_str = '4'
        obj_cnt_list[4] += 1
    elif shape_obj['label'] in ('plate', 'plate+'):
        return
    else:
        prRed('Label {} not support, return'.format(shape_obj['label']))
        return
    one_point_list = shape_obj['points']
    obj_width = float(one_point_list[1][0]) - float(one_point_list[0][0])
    obj_height = float(one_point_list[1][1]) - float(one_point_list[0][1])
    x_center = (float(one_point_list[0][0]) + obj_width/2)/img_width
    y_center = (float(one_point_list[0][1]) + obj_height/2)/img_height
    yolo_width = obj_width/img_width
    yolo_height = obj_height/img_height
    if (x_center <= 0.0) or (y_center <= 0.0) or (yolo_width <= 0.0) or (yolo_height <= 0.0):
        prRed('Yolo pos {} {} {} {} err, return'.format(x_center, y_center, yolo_width, yolo_height))
        return
    fp.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(type_str, x_center, y_center, yolo_width, yolo_height))
    return


def deal_one_image_label_files(
        class_num, 
        train_fp, 
        val_fp, 
        img_file:str, 
        label_file:str, 
        output_dir:str, 
        deal_cnt:int, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    save_image_dir = None
    save_label_dir = None
    save_fp = None
    if ((deal_cnt % 10) >= 8):
        save_fp = val_fp
        save_image_dir = os.path.join(output_dir, "val/images")
        save_label_dir = os.path.join(output_dir, "val/labels")
    else:
        save_fp = train_fp
        save_image_dir = os.path.join(output_dir, "train/images")
        save_label_dir = os.path.join(output_dir, "train/labels")
    dir_name, full_file_name = os.path.split(img_file)
    sub_dir_name0, sub_dir_name1 = dir_name.split('/')[-2], dir_name.split('/')[-1]
    #print(sub_dir_name0, sub_dir_name1)
    save_file_name = sub_dir_name0 + "_" + sub_dir_name1 + "_" + os.path.splitext(full_file_name)[0] + "_" + str(random.randint(0, 999999999999)).zfill(12)
    #print(save_file_name)
    img = cv2.imread(img_file)
    (img_height, img_width, _) = img.shape
    resave_file = os.path.join(save_image_dir, save_file_name + os.path.splitext(full_file_name)[-1])
    os.symlink(img_file, resave_file)
    #shutil.copyfile(img_file, resave_file)
    save_fp.write(resave_file + '\n')
    save_fp.flush()
    #------
    with open(os.path.join(save_label_dir, save_file_name + ".txt"), "w") as fp:
        with open(label_file, 'r') as load_f:
            json_data = json.load(load_f)
            #json_data = json.load(load_f, encoding='utf-8')
            shapes_objs = json_data['shapes']
            for shape_obj in shapes_objs:
                if class_num == 4:
                    labelme2_class4_yolo_data(fp, shape_obj, obj_cnt_list, img_width, img_height)
                elif class_num == 5:
                    labelme2_class5_yolo_data(fp, shape_obj, obj_cnt_list, img_width, img_height)
    return


def deal_dir_files(
        loop_cnt:int,
        class_num, 
        img_label_list:str, 
        output_dir:str, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    #print(img_label_list)
    train_fp = open(output_dir + "/train.txt", "a+")
    val_fp = open(output_dir + "/val.txt", "a+")
    for one_lop in range(loop_cnt):
        prYellow('Loop count{}'.format(one_lop))
        pbar = enumerate(img_label_list)
        pbar = tqdm(pbar, total=len(img_label_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
        for (i, img_label) in pbar:
            img_file = img_label[0]
            label_file = img_label[1]
            img_file_name, _ = os.path.splitext( os.path.split(img_file)[1] )
            label_file_name, _ = os.path.splitext( os.path.split(label_file)[1] )
            if img_file_name != label_file_name:
                sys.stdout.write('\r>> {}: Image file {} and label file {} not fit err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file, label_file))
                sys.stdout.flush()
                os._exit(2)
            deal_one_image_label_files(class_num, train_fp, val_fp, img_file, label_file, output_dir, i, output_size, obj_cnt_list)
    train_fp.close()
    val_fp.close()
    return


def show_statistic_info(input_dir:str, labels_list)->None:
    prYellow('\nShow directory \'{}\' statistic info\n'.format(input_dir))
    labels_cnt_dict = {}
    pbar = enumerate(labels_list)
    pbar = tqdm(pbar, total=len(labels_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (i, label_file) in pbar:
        with open(label_file, 'r') as load_f:
            json_data = json.load(load_f)
            #json_data = json.load(load_f, encoding='utf-8')
            shapes_objs = json_data['shapes']
            for shape_obj in shapes_objs:
                if shape_obj['label'] in labels_cnt_dict:
                    labels_cnt_dict[ shape_obj['label'] ] += 1
                else:
                    labels_cnt_dict[ shape_obj['label'] ] = 0
    #print( len(labels_cnt_dict) )
    # print result
    #print("\n")
    for label in labels_cnt_dict:
        print("%10s " %(label), end='')
    print("%10s" %('total'))
    total_cnt = 0
    for label in labels_cnt_dict:
        print("%10d " %(labels_cnt_dict[ label ]), end='')
        total_cnt += labels_cnt_dict[ label ]
    print("%10d" %(total_cnt))
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    #------
    imgs_list = []
    labels_list = []
    for root, dirs, files in os.walk(args.input_dir, followlinks=True):
        for one_file in sorted(files):
            #print(os.path.splitext(one_file)[-1])
            if os.path.splitext(one_file)[-1] == '.json':
                labels_list.append( os.path.join(root, one_file) )
            else:
                imgs_list.append( os.path.join(root, one_file) )
    #print(len(imgs_list), len(labels_list))
    if len(labels_list) != len(imgs_list):
        sys.stdout.write('\r>> {}: File len {}:{} err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(labels_list), len(imgs_list)))
        sys.stdout.flush()
        os._exit(2)
    img_label_list = []
    for i in range( len(imgs_list) ):
        img_label = []
        img_label.append(imgs_list[i])
        img_label.append(labels_list[i])
        img_label_list.append(img_label[:])
    random.shuffle(img_label_list)
    #print(len(img_label_list))
    #------
    if args.show_statistic:
        show_statistic_info(args.input_dir, labels_list)
        return
    #------
    if (args.output_dir is None) or (args.class_num is None):
        prRed('Not input output_dir or class_num parameter, return')
        return
    output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
    #------
    categories_list = None
    if args.class_num == 4:
        categories_list = categories4_list
    elif args.class_num == 5:
        categories_list = categories5_list
    else:
        prRed('Class num {} err, return'.format(args.class_num))
        return
    obj_cnt_list = [ 0 for _ in range( len(categories_list) ) ]
    deal_dir_files(args.loop_cnt, args.class_num, img_label_list, args.output_dir, output_size, obj_cnt_list)
    # print result
    print("\n")
    for category in categories_list:
        print("%10s " %(category), end='')
    print("%10s" %('total'))
    for i in range( len(categories_list) ):
        print("%10d " %(obj_cnt_list[i]), end='')
    print("%10d" %(sum(obj_cnt_list)))
    #print("\n")
    sys.stdout.write('\r>> {}: Generate yolov dataset success, save dir:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.output_dir))
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main_func()