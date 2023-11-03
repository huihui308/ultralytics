# -*- coding: utf-8 -*-
#
# $ python3 add_plate_to_yolo.py --plate_dir=/home/david/dataset/lpd_lpr/detect_plate_datasets --result_dir=./
# $ python3 add_plate_to_yolo.py --plate_dir=/home/david/dataset/detect/CBD --result_dir=./
#
import cv2
import json
import numpy as np
from tqdm import tqdm
from labelme import utils
from typing import List, OrderedDict
import os, sys, json, time, random, signal, shutil, datetime, argparse


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
    sys.stdout.write('\r>> {}: catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
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
def split_plate(save_main_dir:str, header_file)->None:
    global g_crop_val_cnt
    global g_crop_train_cnt
    train_fp = open(os.path.join(save_main_dir, 'train.txt'), "a+")
    val_fp = open(os.path.join(save_main_dir, 'val.txt'), "a+")
    #------
    time_str = str( datetime.datetime.now().strftime("%Y%m%d_%2H%2M%2S") ) + '_'
    if random.random() < 0.7:
        obj_img_path = save_main_dir + "/train/images/vehicle_" + time_str + str(g_crop_train_cnt).zfill(8) + ".jpg"
        obj_txt_path = save_main_dir + "/train/labels/vehicle_" + time_str + str(g_crop_train_cnt).zfill(8) + ".txt"
        g_crop_train_cnt += 1
        train_fp.write(obj_img_path + '\n')
        train_fp.flush()
    else:
        obj_img_path = save_main_dir + "/val/images/vehicle_" + time_str + str(g_crop_val_cnt).zfill(8) + ".jpg"
        obj_txt_path = save_main_dir + "/val/labels/vehicle_" + time_str + str(g_crop_val_cnt).zfill(8) + ".txt"
        g_crop_val_cnt += 1
        val_fp.write(obj_img_path + '\n')
        val_fp.flush()
    os.symlink(header_file + '.jpg', obj_img_path)
    os.symlink(header_file + '.txt', obj_txt_path)
    train_fp.close()
    val_fp.close()
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
    pbar = enumerate(label_file_list)
    pbar = tqdm(pbar, total=len(label_file_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    print(label_file_list[0])
    for (i, header_file) in pbar:
        split_plate(save_main_dir, header_file)
    return


if __name__ == '__main__':
    main_func()