# -*- coding: utf-8 -*-
# rm -rf output/out.jpg;python3 check_yolov5_data.py --img_file=./input/00289511494253-90_87-258\&505_366\&543-358\&541_264\&543_262\&511_356\&509-0_0_7_29_31_24_13-78-18.jpg --txt_file=./input/00289511494253-90_87-258\&505_366\&543-358\&541_264\&543_262\&511_356\&509-0_0_7_29_31_24_13-78-18.txt;sz output/out.jpg
#
# python3 check_yolo_plate_data.py --result_dir=./
#
import cv2
from pathlib import Path
from typing import List, OrderedDict
import os, sys, json, time, random, signal, shutil, datetime, argparse


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
    parser = argparse.ArgumentParser(description = 'Check yolov data')
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "./output",
        required = False,
        help = "Save plate images and labelme json files dir."
    )
    """
    parser.add_argument(
        "--txt_file",
        type = str,
        #default = "./output",
        required = True,
        help = "Json file."
    )
    """
    return parser.parse_args(args)


def get_label_file(img_file)->None:
    (imgs_file_path, img_name) = os.path.split(img_file.strip('\n'))
    print(imgs_file_path, img_name)
    path_list = imgs_file_path.split('/')
    path_list[-1] = 'labels'
    labels_file_path = '/'
    for path in path_list:
        labels_file_path = os.path.join(labels_file_path, path)
    print(labels_file_path)
    (file_name, _) = os.path.splitext(img_name)
    label_file = os.path.join(labels_file_path, file_name + '.txt')
    print(label_file)
    return label_file


def main_func(args = None)->None:
    args = parse_args(args)
    print(args)
    save_main_dir = os.path.abspath(args.result_dir)
    prYellow('\r>> {}: Save data dir:{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_main_dir))
    if not os.path.exists(save_main_dir):
        prRed('\r>> {}: Save data dir:{} not exist, exit'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_main_dir))
        os._exit(0)
    #train_fp = open(os.path.join(save_main_dir, 'train.txt'), "r")
    #val_fp = open(os.path.join(save_main_dir, 'val.txt'), "r")
    lop_cnt = 0
    for img_file in open(os.path.join(save_main_dir, 'train.txt'), "r"):
        img_file = img_file.strip('\n')
        if not os.path.isfile(img_file):
            prRed('\r>> {}: File {} not exist, continue'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_file))
            continue
        label_file = get_label_file(img_file)
        if not os.path.isfile(label_file):
            prRed('\r>> {}: File {} not exist, continue'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label_file))
            continue
        #-----
        with open(Path(label_file)) as txt_fp:
            txt_data = txt_fp.read()
        #print(txt_data)
        pos_list = txt_data.split(' ')
        print(pos_list)
        print(img_file)
        img = cv2.imread(img_file)
        print(img.shape)
        center_x = int( ( (float)(pos_list[2]) ) *img.shape[1] )
        center_y = int( ( (float)(pos_list[3]) ) *img.shape[0] )
        plate_w = int( ( (float)(pos_list[4]) ) *img.shape[1] )
        plate_h = int( ( (float)(pos_list[5]) ) *img.shape[0] )
        print(center_x, center_y, plate_w, plate_h)
        #------
        ptLeftTop = (center_x - plate_w//2, center_y - plate_h//2)
        ptRightBottom = (center_x + plate_w//2, center_y + plate_h//2)
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        print(ptLeftTop, ptRightBottom)
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        #--------
        point_size = 1
        thickness = 4 # 可以为 0 、4、8
        # 要画的点的坐标
        #points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]
        #for point in points_list:
        #    cv.circle(img, point, point_size, point_color, thickness)
        point = (int( ( (float)(pos_list[6]) ) *img.shape[1] ), int( ( (float)(pos_list[7]) ) *img.shape[0] ))
        point_color = (255, 0, 0) # BGR
        cv2.circle(img, point, point_size, point_color, thickness)
        point = (int( ( (float)(pos_list[8]) ) *img.shape[1] ), int( ( (float)(pos_list[9]) ) *img.shape[0] ))
        point_color = (0, 255, 0) # BGR
        cv2.circle(img, point, point_size, point_color, thickness)
        point = (int( ( (float)(pos_list[10]) ) *img.shape[1] ), int( ( (float)(pos_list[11]) ) *img.shape[0] ))
        point_color = (0, 0, 255) # BGR
        cv2.circle(img, point, point_size, point_color, thickness)
        point = (int( ( (float)(pos_list[12]) ) *img.shape[1] ), int( ( (float)(pos_list[13]) ) *img.shape[0] ))
        point_color = (0, 0, 0) # BGR
        cv2.circle(img, point, point_size, point_color, thickness)
        #------
        cv2.imwrite("./output/out%d.jpg" %(lop_cnt), img)
        lop_cnt += 1
        if lop_cnt == 30:
            break
    return


if __name__ == '__main__':
    signal.signal(signal.SIGINT, term_sig_handler)
    main_func()