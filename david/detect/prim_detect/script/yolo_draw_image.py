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
# $ python3 yolo_draw_image.py --class_num=5 --dataset_dir=./
# $ python3 yolo_draw_image.py --class_num=7 --dataset_dir=/home/david/dataset/detect/yolov8/classes7
# $ python3 yolo_draw_image.py --class_num=11 --dataset_dir=/home/david/dataset/detect/yolov8/classes7
# The default output directory is current ./output
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


def prRed(skk): print("\033[91m \r>> {}: {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prGreen(skk): print("\033[92m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prYellow(skk): print("\033[93m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightPurple(skk): print("\033[94m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prPurple(skk): print("\033[95m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prCyan(skk): print("\033[96m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightGray(skk): print("\033[97m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prBlack(skk): print("\033[98m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))


def term_sig_handler(signum, frame)->None:
    prRed('\n\n\n***************************************\n')
    prRed('catched singal: {}\n'.format(signum))
    prRed('\n***************************************\n')
    sys.stdout.flush()
    os._exit(0)


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Prepare resized images/labels dataset for LPD')
    parser.add_argument(
        "--dataset_dir",
        type = str,
        required = True,
        help = "Input directory to OpenALPR's benchmark end2end us license plates."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./output",
        required = False,
        help = "Save result directory."
    )
    parser.add_argument(
        "--class_num",
        type = int,
        required = True,
        help = "Class num. 4:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3'}, 5:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3', 'lg':'4'}, 6:{'person':'0', 'rider':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}, 7:{'person', 'rider', 'tricycle', 'car', 'R', 'G', 'Y'}, 11:{'person':'0', 'bicycle':'1', 'motorbike':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}"
    )
    parser.add_argument(
        "--parse_cnt",
        type = int,
        default = 20,
        required = False,
        help = "Parse count."
    )
    return parser.parse_args(args)


def get_label_file(img_file)->None:
    (imgs_file_path, img_name) = os.path.split(img_file.strip('\n'))
    #print(imgs_file_path, img_name)
    path_list = imgs_file_path.split('/')
    path_list[-1] = 'labels'
    labels_file_path = '/'
    for path in path_list:
        labels_file_path = os.path.join(labels_file_path, path)
    #print(labels_file_path)
    (file_name, _) = os.path.splitext(img_name)
    label_file = os.path.join(labels_file_path, file_name + '.txt')
    #print(label_file)
    return label_file


def draw_rectangel_to_image(class_num, output_dir, img_file, label_file)->None:
    img_name, _ = os.path.splitext( os.path.split(img_file)[1] )
    #print(img_name)
    image = cv2.imread(img_file)
    (height, width, _) = image.shape
    resave_file = os.path.join(output_dir, img_name + ".jpg")
    #print(resave_file)
    numClassDict4 = {'0':'person', '1':'rider', '2':'car', '3':'lg'}
    numClassDict5 = {'0':'person', '1':'rider', '2':'tricycle', '3':'car', '4':'lg'}
    numClassDict6 = {'0':'person', '1':'rider', '2':'car', '3':'R', '4':'G', '5':'Y'}
    numClassDict7 = {'0':'person', '1':'rider', '2':'tricycle', '3':'car', '4':'R', '5':'G', '6':'Y'}
    numClassDict11 = {'0':'person', '1':'bicycle', '2':'motorbike', '3':'tricycle', '4':'car', '5':'bus', '6':'truck', '7':'plate', '8':'R', '9':'G', '10':'Y'}
    colourList = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (100, 0, 255), (255, 100, 0), (80, 255, 0), (100, 100, 255), (255, 80, 0), (80, 255, 80)]
    for one_line in open(label_file):
        #print(one_line)
        valList = one_line.split(' ')
        type_str = None
        if class_num == 4:
            type_str = numClassDict4[valList[0]]
        elif class_num == 5:
            type_str = numClassDict5[valList[0]]
        elif class_num == 6:
            type_str = numClassDict6[valList[0]]
        elif class_num == 7:
            type_str = numClassDict7[valList[0]]
        elif class_num == 11:
            type_str = numClassDict11[valList[0]]
        xCenter = int(float(valList[1])*width)
        yCenter = int(float(valList[2])*height)
        yoloWidth = int(float(valList[3])*width)
        yoloHeight = int(float(valList[4])*height)
        #left = xCenter - yoloWidth // 2
        #top = yCenter - yoloHeight // 2
        start_point = (xCenter - yoloWidth // 2, yCenter - yoloHeight // 2)
        #right = xCenter + yoloWidth // 2
        #bottom = yCenter + yoloHeight // 2
        end_point = (xCenter + yoloWidth // 2, yCenter + yoloHeight // 2)
        color = colourList[int(valList[0])]
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        image = cv2.putText(image, type_str, start_point, font, 
                   fontScale, color, thickness - 1, cv2.LINE_AA)
        cv2.imwrite(resave_file, image)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    #output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('Save data dir:{}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        prRed('Save data dir:{} not exist, make it'.format(args.output_dir))
        #os._exit(0)
        os.mkdir(args.output_dir)
    # draw images
    random.seed(255)
    txt_list = ['train.txt', 'val.txt']
    for txt_file in txt_list:
        with open(os.path.join(args.dataset_dir, txt_file), "r") as fp:
            lines = fp.readlines()
            #print(len(lines), type(lines))
            for _ in range(args.parse_cnt):
                img_file = lines[random.randint(0, len(lines))].strip('\n')
                prGreen('Deal img_file: {}'.format(img_file))
                if not os.path.isfile(img_file):
                    prRed('File {} not exist, continue'.format(img_file))
                    continue
                label_file = get_label_file(img_file)
                if not os.path.isfile(label_file):
                    prRed('File {} not exist, continue'.format(label_file))
                    continue
                draw_rectangel_to_image(args.class_num, args.output_dir, img_file, label_file)
    return


if __name__ == "__main__":
    main_func()
