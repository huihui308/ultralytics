#
# cmd:
#       rm -rf output_class26;python3 mat2_yolo.py --mat_file=/home/david/dataset/classification/people/annotation.mat --images_dir=/home/david/dataset/classification/people/release_data/release_data --output_dir=./output_class26
#
import os
import scipy
import scipy.io
import pandas as pd
from tqdm import tqdm
from typing import List
from easydict import EasyDict
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


def make_ouput_dir(output_dir:str, labels_dir_list)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lopDir0 in ("train", "val"):
        firstDir = os.path.join(output_dir, lopDir0)
        if not os.path.exists(firstDir):
            #shutil.rmtree(firstDir)
            os.makedirs(firstDir)
        for lopDir1 in labels_dir_list:
            secondDir = os.path.join(firstDir, lopDir1)
            if not os.path.exists(secondDir):
                os.makedirs(secondDir)
    return


def get_labels(pa100k_data, labels_list, labels_dir_list)->None:
    subdata = pa100k_data['attributes']
    df_data = pd.DataFrame(subdata)
    #print(df_data)
    #print("----------------------\n")
    for index, row in df_data.items():
        #print(type(row))
        for (i, dir) in row.items():
            #print(len(dir), dir[0])
            labels_list.append(dir[0])
            labels_dir_list.append('class' + str(i).zfill(4))
            #labels_dir_list.append('class' + str(i).zfill(2) + "_" + dir[0])
    return


def parse_train_data(
        images_type:str,
        label_type:str,
        save_type:str,
        images_dir, 
        output_dir, 
        pa100k_data
)->None:
    images_list = []
    labels_list = []
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
        labels_list.append(row.tolist())
    #print(labels_list[-1])
    #print(len(images_list), len(labels_list))
    if len(images_list) != len(labels_list):
        prRed('images_list {} != labels_list {} err'.format(len(images_list), len(labels_list)))
        return
    pbar = enumerate(labels_list)
    pbar = tqdm(pbar, total=len(labels_list), desc="Processing {}".format(save_type), colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (index, label) in pbar:
        if len(label) != 26:
            prRed('len(label) length {} err, not equal 26'.format(len(label)))
            continue
        image_name = str(images_list[index][0])
        src_image = os.path.join(images_dir, image_name)
        for (i, one_label) in enumerate(label):
            if one_label == 0:
                continue
            save_file_name = os.path.splitext(image_name)[0] + "_" + str(random.randint(0, 999999999999)).zfill(12) + ".jpg"
            dst_image = os.path.join(output_dir + "/" + save_type + "/class" + str(i).zfill(4), save_file_name)
            #print(src_image, dst_image)
            os.symlink(src_image, dst_image)
    return


def mat2_yolo(images_dir, output_dir, pa100k_data)->None:
    labels_list = []
    labels_dir_list = []
    get_labels(pa100k_data, labels_list, labels_dir_list)
    make_ouput_dir(output_dir, labels_dir_list)
    #print(labels_list)
    with open(os.path.join(output_dir, "labels.txt"), "w") as fp:
        for label in labels_list:
            fp.write(label + '\n')
    #df_data.to_csv("%s.txt" % key, index=False)
    #parse_train_data('train_images_name', 'train_label', 'train', images_dir, output_dir, pa100k_data)
    parse_train_data('test_images_name', 'test_label', 'train', images_dir, output_dir, pa100k_data)
    #parse_train_data('val_images_name', 'val_label', 'val', images_dir, output_dir, pa100k_data)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    prYellow('output_dir: {}'.format(args.output_dir))
    print(args.mat_file)
    pa100k_data = scipy.io.loadmat(args.mat_file)
    #print(type(pa100k_data))
    #print(pa100k_data)
    """
    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    print(type(train_image_name), type(val_image_name), type(test_image_name))
    train_label, val_label, test_label = pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']
    """
    key_list = ["attributes", "test_images_name", "test_label",
                "train_images_name", "train_label",
                "val_images_name", "val_label"]
    #for key in key_list:
    mat2_yolo(args.images_dir, args.output_dir, pa100k_data)
    return


if __name__ == "__main__":
    main_func()