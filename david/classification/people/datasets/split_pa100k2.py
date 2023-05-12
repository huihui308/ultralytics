#
# cmd:
#       rm -rf output_class26;python3 split_pa100k2.py --mat_file=/home/david/dataset/classification/people/annotation.mat --images_dir=/home/david/dataset/classification/people/release_data/release_data --output_dir=./output_class26
#
"""
- 性别：男、女
- 年龄：小于18、18-60、大于60
- 朝向：朝前、朝后、侧面
- 配饰：眼镜、帽子、无
- 正面持物：是、否
- 包：双肩包、单肩包、手提包
- 上衣风格：带条纹、带logo、带格子、拼接风格
- 下装风格：带条纹、带图案
- 短袖上衣：是、否
- 长袖上衣：是、否
- 长外套：是、否
- 长裤：是、否
- 短裤：是、否
- 短裙&裙子：是、否
- 穿靴：是、否
"""
import os
import scipy
import scipy.io
import pandas as pd
from tqdm import tqdm
from typing import List
from easydict import EasyDict
import os, sys, math, shutil, random, datetime, signal, argparse


TQDM_BAR_FORMAT = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}'
g_labels_name = [ 
    "女性", "少年", "中年", "老年", "长袖", "短袖", "长裤", "短裙", 
    "上红", "上橙", "上黄", "上绿", "上蓝", "上紫", "上粉", "上黑", "上白", "上灰", "上棕", 
    "下红", "下橙", "下黄", "下绿", "下蓝", "下紫", "下粉", "下黑", "下白", "下灰", "下棕"
]



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
    parser = argparse.ArgumentParser(description = 'Convert matlab data to yolo format.')
    parser.add_argument(
        "--mat_file",
        type = str,
        required = True,
        help = "Matlab file path."
    )
    parser.add_argument(
        "--images_dir",
        type = str,
        required = True,
        help = "Ouput directory to resized images/labels."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = True,
        help = "Ouput directory to resized images/labels."
    )
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str, labels_dir_list:List[str])->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for lop_dir1 in labels_dir_list:
            second_dir = os.path.join(first_dir, lop_dir1)
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def get_labels(pa100k_data, labels_list, labels_dir_list)->None:
    subdata = pa100k_data['attributes']
    df_data = pd.DataFrame(subdata)
    #print(df_data)
    for index, row in df_data.items():
        #print(type(row))
        for (i, dir) in row.items():
            #print(len(dir), dir[0])
            labels_list.append(dir[0])
            labels_dir_list.append('class' + str(i).zfill(4))
            #labels_dir_list.append('class' + str(i).zfill(2) + "_" + dir[0])
    return


def generate_data(
        images_type:str,
        label_type:str,
        save_type:str,
        images_dir:str, 
        output_dir:str, 
        pa100k_data
)->None:
    labels_cnt_list = [ 0 for _ in range(len(g_labels_name)) ]
    images_list = []
    labels_val_list = []
    subdata = pa100k_data[images_type]
    df_data = pd.DataFrame(subdata)
    #print(type(df_data))
    for index, row in df_data.items():
        for (i, image) in row.items():
            #print(len(image), image[0])
            images_list.append(image)
    #print(images_list[-1])
    subdata = pa100k_data[label_type]
    df_data = pd.DataFrame(subdata)
    #print(type(df_data))
    for index, row in df_data.iterrows():
        #print(type(row))
        labels_val_list.append(row.tolist())
    #print(labels_val_list[-1])
    #print(len(images_list), len(labels_val_list))
    if len(images_list) != len(labels_val_list):
        prRed('images_list {} != labels_val_list {} err'.format(len(images_list), len(labels_val_list)))
        return
    pbar = enumerate(labels_val_list)
    pbar = tqdm(pbar, total=len(labels_val_list), desc="Processing {:>8}".format(save_type), colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (index, label) in pbar:
        if len(label) != 26:
            prRed('len(label) length {} err, not equal 26'.format(len(label)))
            continue
        image_name = str(images_list[index][0])
        src_image = os.path.join(images_dir, image_name)
        for (i, one_label) in enumerate(label):
            if one_label == 0:
                continue
            save_num = 0
            if i == 0:
                save_num = 0
                labels_cnt_list[save_num] += 1
            elif i == 1:
                save_num = 3
                labels_cnt_list[save_num] += 1
            elif i == 2:
                save_num = 2
                labels_cnt_list[save_num] += 1
            elif i == 3:
                save_num = 1
                labels_cnt_list[save_num] += 1
            elif i == 13:
                save_num = 5
                labels_cnt_list[save_num] += 1
            elif (i == 14) or (i == 21):
                save_num = 4
                labels_cnt_list[save_num] += 1
            elif i == 22:
                save_num = 6
                labels_cnt_list[save_num] += 1
            elif i == 23:
                save_num = 5
                labels_cnt_list[save_num] += 1
            elif i == 24:
                save_num = 7
                labels_cnt_list[save_num] += 1
            else:
                continue
            save_file_name = os.path.splitext(image_name)[0] + "_" + str(random.randint(0, 999999999999)).zfill(12) + ".jpg"
            dst_image = os.path.join(output_dir + "/" + save_type + "/class" + str(save_num).zfill(4), save_file_name)
            #print(src_image, dst_image)
            os.symlink(src_image, dst_image)
    print("\n")
    for one_label in g_labels_name:
        print("{0:^2} ".format(one_label[:4]), end='')
    print("\n")
    for label_val in labels_cnt_list:
        print("{0:^4} ".format(label_val), end='')
    print("\n")
    return


def split_pa100k2(images_dir, output_dir, pa100k_data)->None:
    labels_list = []
    labels_dir_list = []
    get_labels(pa100k_data, labels_list, labels_dir_list)
    make_ouput_dir(output_dir, labels_dir_list)
    #print(labels_list)
    with open(os.path.join(output_dir, "labels.txt"), "w") as fp:
        for label in labels_list:
            fp.write(label + '\n')
    #df_data.to_csv("%s.txt" % key, index=False)
    generate_data('train_images_name', 'train_label', 'train', images_dir, output_dir, pa100k_data)
    generate_data('test_images_name', 'test_label', 'train', images_dir, output_dir, pa100k_data)
    generate_data('val_images_name', 'val_label', 'val', images_dir, output_dir, pa100k_data)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    prYellow('output_dir: {}\n'.format(args.output_dir))
    #print(args.mat_file)
    pa100k_data = scipy.io.loadmat(args.mat_file)
    #print(type(pa100k_data))
    #print(pa100k_data)
    """
    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    print(type(train_image_name), type(val_image_name), type(test_image_name))
    train_label, val_label, test_label = pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']
    key_list = ["attributes", "test_images_name", "test_label",
                "train_images_name", "train_label",
                "val_images_name", "val_label"]
    """
    split_pa100k2(args.images_dir, args.output_dir, pa100k_data)
    return


if __name__ == "__main__":
    main_func()