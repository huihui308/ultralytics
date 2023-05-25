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
# rm -rf ./test_output_class5;python3 argoverse_to_yolo_class5.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/Argoverse-1.1 --output_dir=./test_output_class5
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


categories_list = ['person', 'rider', 'tricycle', 'car', 'lg']
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


def parse_input_args(args = None):
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
    """
    parser.add_argument(
        "--class_num",
        type = int,
        required = True,
        help = "Class num. 4:{'person':'0', 'rider':'1', 'car':'2', 'lg':'3'}, 5:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3', 'lg':'4'}, 6:{'person':'0', 'rider':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}, 11:{'person':'0', 'bicycle':'1', 'motorbike':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}"
    )
    """
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
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for lop_dir1 in ("images", "labels"):
            secondDir = os.path.join(first_dir, lop_dir1)
            if not os.path.exists(secondDir):
                os.makedirs(secondDir)
    return


def deal_one_image_label_files(
        train_fp, 
        val_fp, 
        img_file:str, 
        targets, 
        categories_dict:dict,
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
    sub_dir_list = dir_name.split('/')
    if len(sub_dir_list) < 1:
        prRed('dir_name {} err'.format(dir_name))
        return
    sub_dir_name1 = sub_dir_list[-1]
    #sub_dir_name0, sub_dir_name1 = dir_name.split('/')[-2], dir_name.split('/')[-1]
    #print(#, sub_dir_name1)
    save_file_name = sub_dir_name1 + "_" + os.path.splitext(full_file_name)[0] + "_" + str(random.randint(0, 999999999999)).zfill(12)
    #print(save_file_name)
    img = cv2.imread(img_file)
    (img_height, img_width, _) = img.shape
    resave_file = os.path.join(save_image_dir, save_file_name + ".jpg")
    os.symlink(img_file, resave_file)
    #shutil.copyfile(img_file, resave_file)
    save_fp.write(resave_file + '\n')
    save_fp.flush()
    #------
    with open(os.path.join(save_label_dir, save_file_name + ".txt"), "w") as fp:
        for one_target in targets:
            if one_target[0] not in categories_dict:
                continue
            type_str = None
            if categories_dict[ one_target[0] ] in ('person'):
                type_str = '0'
                obj_cnt_list[0] += 1
            elif categories_dict[ one_target[0] ] in ('bicycle', 'motorcycle'):
                type_str = '1'
                obj_cnt_list[1] += 1
            elif categories_dict[ one_target[0] ] in ('car', 'bus', 'truck'):
                type_str = '3'
                obj_cnt_list[3] += 1
            elif categories_dict[ one_target[0] ] in ('traffic_light'):
                type_str = '4'
                obj_cnt_list[4] += 1
            elif categories_dict[ one_target[0] ] in ('stop_sign'):
                continue
            else:
                prRed('Category {} not support, continue'.format(categories_dict[ one_target[0] ]))
                continue
            #print( one_target[1] )
            obj_left = float( one_target[1][0] )
            obj_top = float( one_target[1][1] )
            obj_width = float( one_target[1][2] )
            obj_height = float( one_target[1][3] )
            x_center = (obj_left + obj_width/2)/img_width
            y_center = (obj_top + obj_height/2)/img_height
            yolo_width = obj_width/img_width
            yolo_height = obj_height/img_height
            fp.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(type_str, x_center, y_center, yolo_width, yolo_height))
    return


def deal_dir_files(
        json_file:str,
        deal_dir:str, 
        output_dir:str, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    train_fp = open(output_dir + "/train.txt", "a+")
    val_fp = open(output_dir + "/val.txt", "a+")
    #------
    categories_dict = {}
    img_id_dict = {}
    with open(json_file, "r") as fp:
        json_data = json.load(fp, encoding='utf-8')
        for one_category in json_data['categories']:
            categories_dict[one_category['id']] = one_category['name']
        print(categories_dict)
        for lop_img in json_data['images']:
            if lop_img['name'] in img_id_dict:
                prRed('Dict already exist key {}'.format(lop_img['name']))
            img_id_dict[lop_img['name']] = lop_img['id']
        #print( len(img_id_dict), img_id_dict[100], deal_dir )
        imgs_list = [ '' for _ in range( len(img_id_dict) ) ]
        for root, dirs, files in os.walk(deal_dir):
            for one_file in files:
                file_name, file_type = os.path.splitext(one_file)
                if file_type not in ('.jpg', '.png', '.bmp'):
                    prRed('File {} format not support'.format(file_type))
                    continue
                #print(file_name)
                if one_file not in img_id_dict:
                    prRed('File {} not in dict'.format(one_file))
                    continue
                imgs_list[ img_id_dict[one_file] ] = os.path.join(root, one_file)
        #print(imgs_list[100])
        boxes_list = [ [] for _ in range( len(img_id_dict) ) ]
        for one_label in json_data['annotations']:
            category_id_box_tuple = (one_label['category_id'], one_label['bbox'])
            boxes_list[one_label['image_id']].append( category_id_box_tuple )
        #print( len(boxes_list[100]) )
        if len(boxes_list) != len(imgs_list):
            prRed('boxes count {} not equal imgs count {}, return'.format(len(boxes_list), len(imgs_list)))
            return
        #print("\n")
        pbar = enumerate(imgs_list)
        pbar = tqdm(pbar, total=len(imgs_list), desc="Processing {0:>15}".format(deal_dir.split('/')[-1]), colour='blue', bar_format=TQDM_BAR_FORMAT)
        for (i, one_img) in pbar:
            if not os.path.exists(one_img):
                prRed('Image file {} not exist, continue'.format(one_img))
                continue
            deal_one_image_label_files(train_fp, val_fp, one_img, boxes_list[i], categories_dict, output_dir, i, output_size, obj_cnt_list)
        #print("\n")
    train_fp.close()
    val_fp.close()
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_input_args(args)
    output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
    obj_cnt_list = [0 for _ in range( len(categories_list) )]
    #deal_dir_list = ['val']
    deal_dir_list = ['train', 'val']
    for lop_dir in deal_dir_list:
        json_file = os.path.join(args.input_dir, lop_dir + '.json')
        deal_dir_files(json_file, os.path.join(args.input_dir, lop_dir), args.output_dir, output_size, obj_cnt_list)
    print("\n")
    for category in categories_list:
        print("%10s " %(category), end='')
    print("%10s" %('total'))
    for i in range( len(categories_list) ):
        print("%10d " %(obj_cnt_list[i]), end='')
    print("%10d" %(sum(obj_cnt_list)))
    print("\n")
    sys.stdout.write('\r>> {}: Generate yolov dataset success, save dir:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.output_dir))
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main_func()