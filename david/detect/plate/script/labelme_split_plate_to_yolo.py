# -*- coding: utf-8 -*-
#
# $ python3 labelme_split_plate_to_yolo.py --labelme_dir=/home/david/dataset/detect/echo_park --result_dir=./
# $ python3 labelme_split_plate_to_yolo.py --labelme_dir=/home/david/dataset/detect/CBD --result_dir=./
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
        "--labelme_dir",
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
            if filename.split('.')[-1] == 'json':
                if os.path.join(parent, filename.split('.')[0]) in imgs_list:
                    label_file_list.append( os.path.join(parent, filename.split('.')[0]) )
    return


# 两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    :param x1: 第一个框的左上角 x 坐标
    :param y1: 第一个框的左上角 y 坐标
    :param w1: 第一幅图中的检测框的宽度
    :param h1: 第一幅图中的检测框的高度
    :param x2: 第二个框的左上角 x 坐标
    :param y2:
    :param w2:
    :param h2:
    :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    '''
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)


def object_contain_plate(obj_x0, obj_y0, obj_x1, obj_y1, plate_x0, plate_y0, plate_x1, plate_y1)->int:
    if (plate_x0 < obj_x0) or (plate_x0 > obj_x1):
        return 0
    if (plate_y0 < obj_y0) or (plate_y0 > obj_y1):
        return 0
    if (plate_x1 < obj_x0) or (plate_x1 > obj_x1):
        return 0
    if (plate_y1 < obj_y0) or (plate_y1 > obj_y1):
        return 0
    if (plate_y0 - obj_y0)  < ((obj_y1 - obj_y0)/3):
        return 0
    return 1

def judge_plate_in_vehicle(veh_pos:List[str], plate_pos:List[str])->bool:
    #print(one_veh['points'], one_plate['points'])
    #print(type(one_veh['points']), type(one_plate['points']))
    if (len(veh_pos) != 2) or (len(veh_pos[0]) != 2) or len(veh_pos[1]) != 2:
        print("veh_pos format error")
        return False
    if (len(plate_pos) != 2) or (len(plate_pos[0]) != 2) or len(plate_pos[1]) != 2:
        print("plate_pos format error")
        return False
    #print( type(veh_pos[0][0]) )
    """
    x1, y1 = veh_pos[0][0], veh_pos[0][1]
    w1, h1 = (veh_pos[1][0] - veh_pos[0][0]), (veh_pos[1][1] - veh_pos[0][1])
    x2, y2 = plate_pos[0][0], plate_pos[0][1]
    w2, h2 = (plate_pos[1][0] - plate_pos[0][0]), (plate_pos[1][1] - plate_pos[0][1])
    iou_val = bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2)
    """
    veh_x0, veh_y0 = (int)(veh_pos[0][0]), (int)(veh_pos[0][1])
    veh_x1, veh_y1 = (int)(veh_pos[1][0]), (int)(veh_pos[1][1])
    plate_x0, plate_y0 = (int)(plate_pos[0][0]), (int)(plate_pos[0][1])
    plate_x1, plate_y1 = (int)(plate_pos[1][0]), (int)(plate_pos[1][1])
    iou_val = object_contain_plate(veh_x0, veh_y0, veh_x1, veh_y1, plate_x0, plate_y0, plate_x1, plate_y1)
    #print(iou_val)
    if iou_val == 0:
        return False
    return True


g_crop_val_cnt = 0
g_crop_train_cnt = 0
def scrop_vehicle_palte(
        save_main_dir:str, 
        img_file:str, 
        veh_pos:List[str], 
        obj_plate, 
        data_info)->None:
    global g_crop_val_cnt
    global g_crop_train_cnt
    #print(save_main_dir)
    plate_pos = obj_plate['points']
    veh_x0, veh_y0 = (int)(veh_pos[0][0]), (int)(veh_pos[0][1])
    veh_x1, veh_y1 = (int)(veh_pos[1][0]), (int)(veh_pos[1][1])
    #w1, h1 = (veh_pos[1][0] - veh_pos[0][0]), (veh_pos[1][1] - veh_pos[0][1])
    plate_x0, plate_y0 = (int)(plate_pos[0][0]), (int)(plate_pos[0][1])
    plate_x1, plate_y1 = (int)(plate_pos[1][0]), (int)(plate_pos[1][1])
    #w2, h2 = (plate_pos[1][0] - plate_pos[0][0]), (plate_pos[1][1] - plate_pos[0][1])
    #print(img_file)
    img = cv2.imread(img_file)
    #print(img.shape)
    cropped_img = img[veh_y0:veh_y1, veh_x0:veh_x1]  # 裁剪坐标为[y0:y1, x0:x1]
    #cropped_img = cv2.resize(cropped_img, (160,160), interpolation = cv2.INTER_AREA)
    #print(cropped_img.shape)
    ptLeftTop = (plate_x0 - veh_x0, plate_y0 - veh_y0)
    ptRightBottom = (plate_x1 - veh_x0, plate_y1 - veh_y0)
    point_color = (0, 255, 0) # BGR
    thickness = 1 
    lineType = 4
    #print(ptLeftTop, ptRightBottom)
    #cv2.rectangle(cropped_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
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
    #prGreen('\r>> {}: Save image:{}, txt:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), obj_img_path, obj_txt_path))
    cv2.imwrite(obj_img_path, cropped_img)
    #------
    crop_length = veh_x1 - veh_x0
    crop_height = veh_y1 - veh_y0
    plate_w = plate_x1 - plate_x0
    plate_h = plate_y1 - plate_y0
    center_x = ptLeftTop[0] + plate_w/2
    center_y = ptLeftTop[1] + plate_h/2
    new_plate_x0 = ptLeftTop[0]
    new_plate_y0 = ptLeftTop[1]
    new_plate_x1 = new_plate_x0 + plate_w
    new_plate_y1 = new_plate_y0
    new_plate_x2 = new_plate_x0 + plate_w
    new_plate_y2 = new_plate_y0 + plate_h
    new_plate_x3 = new_plate_x0
    new_plate_y3 = new_plate_y0 + plate_h
    label_val = 0
    if obj_plate['label'] == 'plate':
        label_val = 0
    elif obj_plate['label'] == 'plate+':
        label_val = 1
    with open(obj_txt_path, 'w') as txt_fp:
        txt_fp.write( '{}  {} {} {} {} {} {} {} {} {} {} {} {}'.format(
            (str)(label_val), 
            (float)(center_x/crop_length),
            (float)(center_y/crop_height),
            (float)(plate_w/crop_length),
            (float)(plate_h/crop_height),
            (float)(new_plate_x0/crop_length),
            (float)(new_plate_y0/crop_height),
            (float)(new_plate_x1/crop_length),
            (float)(new_plate_y1/crop_height),
            (float)(new_plate_x2/crop_length),
            (float)(new_plate_y2/crop_height),
            (float)(new_plate_x3/crop_length),
            (float)(new_plate_y3/crop_height)
        ) )
    return


def split_plate(
        save_main_dir:str, 
        header_file, 
        json_fp, 
        data_info)->None:
    json_data = json_fp.read()
    parsed_json = json.loads(json_data)
    #print(parsed_json['shapes'])
    plate_list = []
    vehicle_list = []
    for one_obj in parsed_json['shapes']:
        if one_obj['label'] in ('plate', 'plate+'):
            plate_list.append(one_obj)
        elif one_obj['label'] in ('bus', 'truck', 'car'):
            vehicle_list.append(one_obj)
    #print(plate_list)
    #print(vehicle_list)
    for one_veh in vehicle_list:
        for one_plate in plate_list:
            if judge_plate_in_vehicle(one_veh['points'], one_plate['points']) == False:
                continue
            scrop_vehicle_palte(save_main_dir, header_file+'.jpg', one_veh['points'], one_plate, data_info)
    return


def main_func(args = None)->None:
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    #------
    save_main_dir = os.path.abspath(args.result_dir)
    prYellow('Save data dir:{}'.format(save_main_dir))
    make_directory(save_main_dir)
    #------
    label_file_list = []
    get_file_list(args.labelme_dir, label_file_list)
    #prYellow('label_file_list len:{}'.format(len(label_file_list)))
    #for header_file in label_file_list:
    train_fp = open(os.path.join(save_main_dir, 'train.txt'), "a+")
    val_fp = open(os.path.join(save_main_dir, 'val.txt'), "a+")
    pbar = enumerate(label_file_list)
    pbar = tqdm(pbar, total=len(label_file_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (i, header_file) in pbar:
        with open(header_file + '.json') as json_fp:
            if (i % 10) < int(args.train_ratio * 10):
                data_info = [train_fp, 0]
            else:
                data_info = [val_fp, 1]
            split_plate(save_main_dir, header_file, json_fp, data_info)
    train_fp.close()
    val_fp.close()
    return


if __name__ == '__main__':
    main_func()