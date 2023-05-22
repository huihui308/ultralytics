#
# cmd:
#       rm -rf tricycle_datasets;python3 convert_dataset.py --input_dir=/home/david/dataset/classification/tricycle/haitian_label_20230518 --output_dir=./tricycle_datasets
#
import cv2
import scipy
import scipy.io
from tqdm import tqdm
from typing import List
from pathlib import Path
from easydict import EasyDict
import os, sys, math, json, shutil, random, datetime, signal, argparse


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
    parser = argparse.ArgumentParser(description = 'Convert dataset to torch format.')
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
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str, class_cnt:int)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for i in range(class_cnt):
            second_dir = os.path.join(first_dir, 'class' + str(i).zfill(4))
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def deal_files(files_list, output_dir, obj_cnt_list)->None:
    pbar = enumerate(files_list)
    pbar = tqdm(pbar, total=len(files_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (lop_cnt, image_file) in pbar:
    #for image_file in files_list:
        file_name, file_type = os.path.splitext(image_file)
        json_file = file_name + '.json'
        if not Path( json_file ).is_file():
            #prRed('{} not exist, continue'.format(json_file))
            continue
        with open(json_file, "r") as fp:
            json_data = json.load(fp, encoding='utf-8')
            #print(json_data['forward']['name'])
            x0, y0, x1, y1 = json_data['locate']
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            img = cv2.imread(image_file)
            cropped_img = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
            #cv2.imwrite("./cv_cut_thor.jpg", cropped_img)
            labels_list = []
            if json_data['forward']['name'] == '有棚':
                labels_list.append('class0000')
                obj_cnt_list[0] += 1
            if json_data['backward']['name'] == '有棚':
                labels_list.append('class0001')
                obj_cnt_list[1] += 1
            if json_data['purpose']['name'] == '快递':
                labels_list.append('class0002')
                obj_cnt_list[2] += 1
            if json_data['color']['name'] == '红色':
                labels_list.append('class0003')
                obj_cnt_list[3] += 1
            elif json_data['color']['name'] == '橙色':
                labels_list.append('class0004')
                obj_cnt_list[4] += 1
            elif json_data['color']['name'] == '黄色':
                labels_list.append('class0005')
                obj_cnt_list[5] += 1
            elif json_data['color']['name'] == '绿色':
                labels_list.append('class0006')
                obj_cnt_list[6] += 1
            elif json_data['color']['name'] == '蓝色':
                labels_list.append('class0007')
                obj_cnt_list[7] += 1
            elif json_data['color']['name'] == '紫色':
                labels_list.append('class0008')
                obj_cnt_list[8] += 1
            elif json_data['color']['name'] == '粉色':
                labels_list.append('class0009')
                obj_cnt_list[9] += 1
            elif json_data['color']['name'] == '黑色':
                labels_list.append('class0010')
                obj_cnt_list[10] += 1
            elif json_data['color']['name'] == '白色':
                labels_list.append('class0011')
                obj_cnt_list[11] += 1
            elif json_data['color']['name'] == '灰色':
                labels_list.append('class0012')
                obj_cnt_list[12] += 1
            elif json_data['color']['name'] == '棕色':
                labels_list.append('class0013')
                obj_cnt_list[13] += 1
            if json_data['purpose']['name'] == '快递':
                labels_list.append('class0014')
                obj_cnt_list[14] += 1
            elif json_data['purpose']['name'] == '外卖':
                labels_list.append('class0015')
                obj_cnt_list[15] += 1
            elif json_data['purpose']['name'] == '货运':
                labels_list.append('class0016')
                obj_cnt_list[16] += 1
            elif json_data['purpose']['name'] == '载人':
                labels_list.append('class0017')
                obj_cnt_list[17] += 1
            if json_data['brand']['name'] == '顺丰':
                labels_list.append('class0018')
                obj_cnt_list[18] += 1
            elif json_data['brand']['name'] == '京东':
                labels_list.append('class0019')
                obj_cnt_list[19] += 1
            elif json_data['brand']['name'] == '邮政':
                labels_list.append('class0020')
                obj_cnt_list[20] += 1
            elif json_data['brand']['name'] == '中通':
                labels_list.append('class0021')
                obj_cnt_list[21] += 1
            elif json_data['brand']['name'] == '圆通':
                labels_list.append('class0022')
                obj_cnt_list[22] += 1
            elif json_data['brand']['name'] == '申通':
                labels_list.append('class0023')
                obj_cnt_list[23] += 1
            elif json_data['brand']['name'] == '德邦':
                labels_list.append('class0024')
                obj_cnt_list[24] += 1
            elif json_data['brand']['name'] == '韵达':
                labels_list.append('class0025')
                obj_cnt_list[25] += 1
            elif json_data['brand']['name'] == '天天':
                labels_list.append('class0026')
                obj_cnt_list[26] += 1
            elif json_data['brand']['name'] == '百世':
                labels_list.append('class0027')
                obj_cnt_list[27] += 1
            elif json_data['brand']['name'] == '丹鸟':
                labels_list.append('class0028')
                obj_cnt_list[28] += 1
            elif json_data['brand']['name'] == '天猫':
                labels_list.append('class0029')
                obj_cnt_list[29] += 1
            elif json_data['brand']['name'] == '海皇':
                labels_list.append('class0030')
                obj_cnt_list[30] += 1
            elif json_data['brand']['name'] == '未知':
                labels_list.append('class0031')
                obj_cnt_list[31] += 1
        #print(len(labels_list), output_dir)
        save_file = None
        save_file_name = file_name.split('/')[-1] + "_" + str(random.randint(0, 999999999999)).zfill(12) + file_type
        if (lop_cnt % 10) >= 8:
            save_file = os.path.join(output_dir, 'val')
        else:
            save_file = os.path.join(output_dir, 'train')
        #save_file = os.path.join(save_file, save_file_name)
        #print(save_file)
        for lop_dir in labels_list:
            tmp_save_file = save_file + '/' + lop_dir + '/' + save_file_name
            #print(tmp_save_file)
            cv2.imwrite(tmp_save_file, cropped_img)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    files_list = []
    obj_cnt_list = [0 for _ in range(32)]
    make_ouput_dir(args.output_dir, len(obj_cnt_list))
    for root, dirs, files in os.walk(args.input_dir):
        for one_file in files:
            file_name, file_type = os.path.splitext(one_file)
            #if file_type != '.json':
            if file_type not in ('.jpg', '.png', '.bmp'):
                continue
            files_list.append( os.path.join(root, one_file) )
    #print(len(files_list), files_list[0])
    deal_files(files_list, args.output_dir, obj_cnt_list)
    print("\n")
    for i in range(len(obj_cnt_list)):
        print("%3s  " %(str(i).zfill(3)), end="")
    print("\n")
    for val in obj_cnt_list:
        print("%3d  " %(val), end="")
    print("\n")
    return


if __name__ == "__main__":
    main_func()