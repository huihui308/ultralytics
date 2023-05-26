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
# python3 bdd_to_yolo.py --class_num=4 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/bdd --output_dir=./output_class4
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


categories4_list = ['person', 'rider', 'car', 'lg']
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


def term_sig_handler(signum, frame)->None:
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
        required = True,
        help = "Ouput directory to resized images/labels."
    )
    parser.add_argument(
        "--class_num",
        type = int,
        required = True,
        help = "Class num. 4:{'person':'0', 'rider':'1', 'car':'2', 'lg':'3'}, 5:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3', 'lg':'4'}, 6:{'person':'0', 'rider':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}, 11:{'person':'0', 'bicycle':'1', 'motorbike':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}"
    )
    parser.add_argument(
        "--target_width",
        type = int,
        required = True,
        help = "Target width for resized images/labels."
    )
    parser.add_argument(
        "--target_height",
        type = int,
        required = True,
        help = "Target height for resized images/labels."
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


def bdd2_class4_yolo_data(fp, lop_obj, obj_cnt_list, img_width, img_height)->None:
    #categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']
    type_str = None
    if lop_obj['category'] in ('person'):
        type_str = '0'
        obj_cnt_list[0] += 1
    elif lop_obj['category'] in ('bike', 'motor', 'rider'):
        type_str = '1'
        obj_cnt_list[1] += 1
    elif lop_obj['category'] in ('car', 'bus', 'truck'):
        type_str = '2'
        obj_cnt_list[2] += 1
    elif lop_obj['category'] in ('traffic light'):
        type_str = '3'
        obj_cnt_list[3] += 1
    elif lop_obj['category'] in ('traffic sign', 'train'):
        return
    else:
        prRed('Category {} not support, return'.format(lop_obj['category']))
        return
    xy = lop_obj['box2d']
    if (xy['x1'] >= xy['x2']) or (xy['y1'] >= xy['y2']):
        return
    #print(xy['x1'], xy['y1'], xy['x2'], xy['y2'], categorys.index(lop_obj['category']))
    obj_width = xy['x2'] - xy['x1']
    obj_height = xy['y2'] - xy['y1']
    x_center = (xy['x1'] + obj_width/2)/img_width
    y_center = (xy['y1'] + obj_height/2)/img_height
    yolo_width = obj_width/img_width
    yolo_height = obj_height/img_height
    fp.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(type_str, x_center, y_center, yolo_width, yolo_height))
    return


def bdd2_class5_yolo_data(fp, lop_obj, obj_cnt_list, img_width, img_height)->None:
    #categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']
    type_str = None
    if lop_obj['category'] in ('person'):
        type_str = '0'
        obj_cnt_list[0] += 1
    elif lop_obj['category'] in ('bike', 'motor', 'rider'):
        type_str = '1'
        obj_cnt_list[1] += 1
    elif lop_obj['category'] in ('car', 'bus', 'truck'):
        type_str = '3'
        obj_cnt_list[3] += 1
    elif lop_obj['category'] in ('traffic light'):
        type_str = '4'
        obj_cnt_list[4] += 1
    elif lop_obj['category'] in ('traffic sign', 'train'):
        return
    else:
        prRed('Category {} not support, return'.format(lop_obj['category']))
        return
    xy = lop_obj['box2d']
    if (xy['x1'] >= xy['x2']) or (xy['y1'] >= xy['y2']):
        return
    #print(xy['x1'], xy['y1'], xy['x2'], xy['y2'], categorys.index(lop_obj['category']))
    obj_width = xy['x2'] - xy['x1']
    obj_height = xy['y2'] - xy['y1']
    x_center = (xy['x1'] + obj_width/2)/img_width
    y_center = (xy['y1'] + obj_height/2)/img_height
    yolo_width = obj_width/img_width
    yolo_height = obj_height/img_height
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
    save_file_name = sub_dir_name0 + "_" + sub_dir_name1 + "_" + os.path.splitext(full_file_name)[0] + "_" + str(random.randint(100000000000, 999999999999)).zfill(12)
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
            jsonData = json.load(load_f)
            #jsonData = json.load(load_f, encoding='utf-8')
            frames = jsonData['frames']
            for frame in frames:
                objects = frame['objects']
                for lop_obj in objects:
                    if 'box2d' not in lop_obj:
                        continue
                    if class_num == 4:
                        bdd2_class4_yolo_data(fp, lop_obj, obj_cnt_list, img_width, img_height)
                    elif class_num == 5:
                        bdd2_class5_yolo_data(fp, lop_obj, obj_cnt_list, img_width, img_height)
    return


def deal_dir_files(
        class_num, 
        img_label_list:str, 
        output_dir:str, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    #print(len(img_label_list))
    train_fp = open(output_dir + "/train.txt", "a+")
    val_fp = open(output_dir + "/val.txt", "a+")
    pbar = enumerate(img_label_list)
    pbar = tqdm(pbar, total=len(img_label_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (i, img_label) in pbar:
        img_file = img_label[0]
        label_file = img_label[1]
        img_file_name, _ = os.path.splitext( os.path.split(img_file)[1] )
        label_file_name, _ = os.path.splitext( os.path.split(label_file)[1] )
        if img_file_name != label_file_name:
        #if os.path.splitext(img_file)[0] != os.path.splitext(label_file)[0]:
            sys.stdout.write('\r>> {}: Image file {} and label file {} not fit err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file, label_file))
            sys.stdout.flush()
            os._exit(2)
        deal_one_image_label_files(class_num, train_fp, val_fp, img_file, label_file, output_dir, i, output_size, obj_cnt_list)
    train_fp.close()
    val_fp.close()
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
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
    categories_list = None
    if args.class_num == 4:
        categories_list = categories4_list
    elif args.class_num == 5:
        categories_list = categories5_list
    else:
        prRed('Class num {} err, return'.format(args.class_num))
        return
    obj_cnt_list = [ 0 for _ in range( len(categories_list) ) ]
    deal_dir_files(args.class_num, img_label_list, args.output_dir, output_size, obj_cnt_list)
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