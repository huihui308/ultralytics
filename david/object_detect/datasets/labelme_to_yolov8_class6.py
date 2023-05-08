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
# python3 labelme_to_yolov8_type6.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output
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
        required = True,
        help = "Ouput directory to resized images/labels."
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
    save_file_name = sub_dir_name + "_" + str(random.randint(10000000, 99999999)).zfill(8)
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


def generate_yolo_dataset(
        train_fp, 
        val_fp, 
        img_file:str, 
        label_dict_list, 
        output_dir:str, 
        deal_cnt:int, 
        output_size:List[int]
)->None:
    """ Create Yolo dataset. """
    #sys.stdout.write('\r>> {}: Deal file {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file))
    #sys.stdout.flush()
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
    sub_dir_name = dir_name.split('/')[-1]
    save_file_name = sub_dir_name + "_" + str(random.randint(10000000, 99999999)).zfill(8)
    #print( save_file_name )
    # resize labels
    #w, h = output_size
    img = cv2.imread(img_file)
    (height, width, _) = img.shape
    #ratio_w = float( float(w)/float(width) )
    #ratio_h = float( float(h)/float(height) )
    # resize images
    resave_file = os.path.join(save_image_dir, save_file_name + ".jpg")
    #image = Image.open(img_file)
    #image.save(os.path.join(save_image_dir, save_file_name + ".jpg"))
    #print(type(img_file))
    shutil.copyfile(img_file, resave_file)
    save_fp.write(resave_file + '\n')
    save_fp.flush()
    classNumDict = {'person':'0', 'ride':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}
    with open(os.path.join(save_label_dir, save_file_name + ".txt"), "w") as f:
        for obj_dict in label_dict_list:
            for label_str, point_list in obj_dict.items():
                #print(label_str)
                #print(classNumDict[label_str])
                if len(point_list) == 0:
                    continue
                for one_point_list in point_list:
                    #print(one_point_list)
                    if len(one_point_list) != 2:
                        sys.stdout.write('\r>> {}: Label file point len err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        sys.stdout.flush()
                        os._exit(2)
                    objWidth = float(one_point_list[1][0]) - float(one_point_list[0][0])
                    objHeight = float(one_point_list[1][1]) - float(one_point_list[0][1])
                    xCenter = (float(one_point_list[0][0]) + objWidth/2)/width
                    yCenter = (float(one_point_list[0][1]) + objHeight/2)/height
                    yoloWidth = objWidth/width
                    yoloHeight = objHeight/height
                    if (xCenter <= 0.0) or (yCenter <= 0.0) or (yoloWidth <= 0.0) or (yoloHeight <= 0.0):
                        continue
                    #x1 = float(one_point_list[0][0]) / width
                    #y1 = float(one_point_list[0][1]) / height
                    #x2 = float(one_point_list[1][0]) / width
                    #y2 = float(one_point_list[1][1]) / height
                    f.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(classNumDict[label_str], xCenter, yCenter, yoloWidth, yoloHeight))
    return


def deal_one_image_label_files(
        train_fp, 
        val_fp, 
        img_file:str, 
        label_file:str, 
        output_dir:str, 
        deal_cnt:int, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    with open(label_file, 'r') as load_f:
        load_dict = json.load(load_f)
        shapes_objs = load_dict['shapes']
        person_list = []
        ride_list = []
        car_list = []
        R_list = []
        G_list = []
        Y_list = []
        for shape_obj in shapes_objs:
            if shape_obj['label'] == 'person':
                obj_cnt_list[0] += 1
                if len(shape_obj['points']) != 2:
                    sys.stdout.write('\r>> {}: Label file point format err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    sys.stdout.flush()
                    os._exit(2)
                person_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'bicycle':
                obj_cnt_list[1] += 1
                ride_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'motorbike':
                obj_cnt_list[1] += 1
                ride_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'tricycle':
                obj_cnt_list[1] += 1
                ride_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'car':
                obj_cnt_list[2] += 1
                car_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'bus':
                obj_cnt_list[2] += 1
                car_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'truck':
                obj_cnt_list[2] += 1
                car_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'R':
                obj_cnt_list[3] += 1
                R_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'G':
                obj_cnt_list[4] += 1
                G_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'Y':
                obj_cnt_list[5] += 1
                Y_list.append( shape_obj['points'] )
        label_dict_list = []
        label_dict_list.append( {'person': person_list } )
        label_dict_list.append( {'ride': ride_list} )
        label_dict_list.append( {'car': car_list} )
        label_dict_list.append( {'R': R_list} )
        label_dict_list.append( {'G': G_list} )
        label_dict_list.append( {'Y': Y_list} )
        #print( label_dict_list )
        generate_yolo_dataset(train_fp, val_fp, img_file, label_dict_list, output_dir, deal_cnt, output_size)
        #GenerateKITTIDataset(img_file, label_dict_list, output_dir, deal_cnt, output_size)
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


def deal_dir_files(deal_dir:str, output_dir:str, output_size:List[int], obj_cnt_list)->None:
    img_list = []
    label_list = []
    for root, dirs, files in os.walk(deal_dir):
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
    train_fp = open(output_dir + "/train.txt", "a+")
    val_fp = open(output_dir + "/val.txt", "a+")
    pbar = enumerate(img_label_list)
    pbar = tqdm(pbar, total=len(img_label_list), desc="Processing {0:>15}".format(deal_dir.split('/')[-1]), colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (i, img_label) in pbar:
        img_file = img_label[0]
        label_file = img_label[1]
        if os.path.splitext(img_file)[0] != os.path.splitext(label_file)[0]:
            sys.stdout.write('\r>> {}: Image file {} and label file {} not fit err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file, label_file))
            sys.stdout.flush()
            os._exit(2)
        deal_one_image_label_files(train_fp, val_fp, img_file, label_file, output_dir, i, output_size, obj_cnt_list)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
    obj_cnt_list = [0 for _ in range(6)]
    for root, dirs, files in os.walk(args.input_dir):
        for dir in dirs:
            deal_dir_files(os.path.join(root, dir), args.output_dir, output_size, obj_cnt_list)
    print("\n%10s %10s %10s %10s %10s %10s %10s" %('person', 'ride', 'car', 'R', 'G', 'Y', 'total'))
    print("%10d %10d %10d %10d %10d %10d %10d\n" %(obj_cnt_list[0], obj_cnt_list[1], obj_cnt_list[2], obj_cnt_list[3], obj_cnt_list[4], obj_cnt_list[5], sum(obj_cnt_list)))
    sys.stdout.write('\r>> {}: Generate yolov dataset success, save dir:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.output_dir))
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main_func()