# -*- coding: utf-8 -*-
# 
# This file soft link plate jpg and txt file to result_dir. So if you data is CCPD, you must use ccpd_process.py to generate txt files.
# $ python3 add_plate_to_yolo.py --plate_dir=/home/david/dataset/lpd_lpr/detect_plate_datasets --result_dir=./
#
import cv2
import json
import numpy as np
from tqdm import tqdm
from labelme import utils
from typing import List, OrderedDict
import os, sys, json, time, random, signal, shutil, datetime, argparse


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
    prRed('catched singal: {}\n'.format(signum))
    sys.stdout.flush()
    os._exit(0)


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Split plate from labelme data')
    parser.add_argument(
        "--plate_dir",
        type = str,
        required = True,
        help = "Labelme dir."
    )
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "./output",
        required = False,
        help = "Save plate images and labelme json files dir."
    )
    parser.add_argument(
        "--train_ratio",
        type = float,
        default = 0.8,
        required = False,
        help = "Train ratio, the other is val."
    )
    return parser.parse_args(args)


def make_directory(output_dir:str)->None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists( os.path.join(output_dir, 'train/images') ):
        os.makedirs( os.path.join(output_dir, 'train/images') )
    if not os.path.exists( os.path.join(output_dir, 'train/labels') ):
        os.makedirs( os.path.join(output_dir, 'train/labels') )
    if not os.path.exists( os.path.join(output_dir, 'val/images') ):
        os.makedirs( os.path.join(output_dir, 'val/images') )
    if not os.path.exists( os.path.join(output_dir, 'val/labels') ):
        os.makedirs( os.path.join(output_dir, 'val/labels') )
    return


def get_file_list(input_dir: str, label_file_list:List[str])->None:
    imgs_list = []
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'jpg':
                imgs_list.append( os.path.join(parent, filename.split('.')[0]) )
    #print(imgs_list)
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'txt':
                if os.path.join(parent, filename.split('.')[0]) in imgs_list:
                    label_file_list.append( os.path.join(parent, filename.split('.')[0]) )
    return


g_crop_val_cnt = 0
g_crop_train_cnt = 0
def split_plate(save_main_dir:str, header_file, data_info)->None:
    global g_crop_val_cnt
    global g_crop_train_cnt
    #------
    time_str = str( datetime.datetime.now().strftime("%Y%m%d_%2H%2M%2S") ) + '_'
    if data_info[1] == 0:
        obj_img_path = save_main_dir + "/train/images/vehicle_" + time_str + str(g_crop_train_cnt).zfill(8) + ".jpg"
        obj_txt_path = save_main_dir + "/train/labels/vehicle_" + time_str + str(g_crop_train_cnt).zfill(8) + ".txt"
        g_crop_train_cnt += 1
        data_info[0].write(obj_img_path + '\n')
        data_info[0].flush()
    else:
        obj_img_path = save_main_dir + "/val/images/vehicle_" + time_str + str(g_crop_val_cnt).zfill(8) + ".jpg"
        obj_txt_path = save_main_dir + "/val/labels/vehicle_" + time_str + str(g_crop_val_cnt).zfill(8) + ".txt"
        g_crop_val_cnt += 1
        data_info[0].write(obj_img_path + '\n')
        data_info[0].flush()
    os.symlink(header_file + '.jpg', obj_img_path)
    os.symlink(header_file + '.txt', obj_txt_path)
    return


def main_func(args = None)->None:
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    #------
    save_main_dir = os.path.abspath(args.result_dir)
    prYellow('\r>> {}: Save data dir:{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_main_dir))
    make_directory(save_main_dir)
    #------
    label_file_list = []
    get_file_list(args.plate_dir, label_file_list)
    #prYellow('\r>> {}: label_file_list len:{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(label_file_list)))
    train_fp = open(os.path.join(save_main_dir, 'train.txt'), "a+")
    val_fp = open(os.path.join(save_main_dir, 'val.txt'), "a+")
    pbar = enumerate(label_file_list)
    pbar = tqdm(pbar, total=len(label_file_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    #print(label_file_list[0])
    for (i, header_file) in pbar:
        if (i % 10) < int(args.train_ratio * 10):
            data_info = [train_fp, 0]
        else:
            data_info = [val_fp, 1]
        split_plate(save_main_dir, header_file, data_info)
    train_fp.close()
    val_fp.close()
    return


if __name__ == '__main__':
    main_func()