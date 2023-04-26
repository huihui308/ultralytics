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
# python3 labelme_to_yolov8.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/cuiwei --output_dir=./output
#
################################################################################

""" Script to prepare resized images/labels for primary detect. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import json
import threading
from PIL import Image
from typing import List
import os, sys, math, shutil, random, datetime, signal, argparse


def TermSigHandler(signum, frame) -> None:
    sys.stdout.write('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.write('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.write('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def ParseArgs(args = None):
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


def MakeOuptDir(output_dir:str)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs( output_dir )
    for lopDir in ("images", "labels"):
        firstDir = os.path.join(output_dir, lopDir)
        if os.path.exists(firstDir):
            shutil.rmtree(firstDir)
        os.makedirs(firstDir)
        os.makedirs( os.path.join(firstDir, "train") )
        os.makedirs( os.path.join(firstDir, "val") )
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


def GenerateYoloDataset(img_file:str, label_dict_list, output_dir:str, deal_cnt:int, output_size:List[int])->None:
    """ Create Yolo dataset. """
    sys.stdout.write('\r>> {}: Deal file {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file))
    sys.stdout.flush()
    save_image_dir = None
    save_label_dir = None
    if ((deal_cnt % 10) >= 8):
        save_image_dir = os.path.join(output_dir, "images/val")
        save_label_dir = os.path.join(output_dir, "labels/val")
    else:
        save_image_dir = os.path.join(output_dir, "images/train")
        save_label_dir = os.path.join(output_dir, "labels/train")
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
    #image = Image.open(img_file)
    #image.save(os.path.join(save_image_dir, save_file_name + ".jpg"))
    #print(img_file)
    shutil.copyfile(img_file, os.path.join(save_image_dir, save_file_name + ".jpg"))
    classNumDict = {'person':'0', 'bicycle':'1', 'motor':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}
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
                    #x1 = float(one_point_list[0][0]) / width
                    #y1 = float(one_point_list[0][1]) / height
                    #x2 = float(one_point_list[1][0]) / width
                    #y2 = float(one_point_list[1][1]) / height
                    f.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(classNumDict[label_str], xCenter, yCenter, yoloWidth, yoloHeight))
    return


def DealOneImageLabelFiles(img_file:str, label_file:str, output_dir:str, deal_cnt:int, output_size:List[int])->None:
    with open(label_file, 'r') as load_f:
        load_dict = json.load(load_f)
        shapes_objs = load_dict['shapes']
        person_list = []
        bicycle_list = []
        motor_list = []
        tricycle_list = []
        car_list = []
        bus_list = []
        truck_list = []
        plate_list = []
        R_list = []
        G_list = []
        Y_list = []
        for shape_obj in shapes_objs:
            if shape_obj['label'] == 'person':
                if len(shape_obj['points']) != 2:
                    sys.stdout.write('\r>> {}: Label file point format err!!!!\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    sys.stdout.flush()
                    os._exit(2)
                person_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'bicycle':
                bicycle_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'motor':
                motor_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'tricycle':
                tricycle_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'car':
                car_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'bus':
                bus_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'truck':
                truck_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'plate':
                plate_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'R':
                R_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'G':
                G_list.append( shape_obj['points'] )
            elif shape_obj['label'] == 'Y':
                Y_list.append( shape_obj['points'] )
        label_dict_list = []
        person_dict = { 'person': person_list }
        label_dict_list.append( person_dict )
        bicycle_dict = { 'bicycle': bicycle_list }
        label_dict_list.append( bicycle_dict )
        motor_dict = { 'motor': motor_list }
        label_dict_list.append( motor_dict )
        tricycle_dict = { 'tricycle': tricycle_list }
        label_dict_list.append( tricycle_dict )
        car_dict = { 'car': car_list }
        label_dict_list.append( car_dict )
        bus_dict = { 'bus': car_list }
        label_dict_list.append( bus_dict )
        truck_dict = { 'truck': car_list }
        label_dict_list.append( truck_dict )
        plate_dict = { 'plate': plate_list }
        label_dict_list.append( plate_dict )
        R_dict = {'R': R_list }
        label_dict_list.append( R_dict )
        G_dict = {'G': G_list }
        label_dict_list.append( G_dict )
        Y_dict = {'Y': Y_list }
        label_dict_list.append( Y_dict )
        #print( label_dict_list )
        GenerateYoloDataset(img_file, label_dict_list, output_dir, deal_cnt, output_size)
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
            DealOneImageLabelFiles(img_file, label_file, self.output_dir, i, self.output_size)
        return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, TermSigHandler)
    args = ParseArgs(args)
    output_size = (args.target_width, args.target_height)
    MakeOuptDir(args.output_dir)
    deal_thread_list = []
    for root, dirs, files in os.walk(args.input_dir):
        for dir in dirs:
            #DealDirFiles(os.path.join(root, dir), args.output_dir, output_size)
            deal_thread = DealDirFilesThread(os.path.join(root, dir), args.output_dir, output_size)
            deal_thread_list.append(deal_thread)
    for one_thread in deal_thread_list:
        one_thread.start()
    for one_thread in deal_thread_list:
        one_thread.join()
    sys.stdout.write('\r>> {}: Generate yolov8 dataset success\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main_func()