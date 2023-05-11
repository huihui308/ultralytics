#
# cmd:
#       rm -rf pa100k;python3 mat2_yolo.py --mat_file=/home/david/dataset/classification/people/annotation.mat --output_dir=./pa100k
#
import os
import scipy
import scipy.io
import pandas as pd
from typing import List
from easydict import EasyDict
import os, sys, math, shutil, random, datetime, signal, argparse


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


def get_labels(pa100k_data, labes_dir_list)->None:
    subdata = pa100k_data['attributes']
    df_data = pd.DataFrame(subdata)
    #print(df_data)
    #print("----------------------\n")
    for index, row in df_data.items():
        #print(type(row))
        for (i, dir) in row.items():
            #print(len(dir), dir[0])
            labes_dir_list.append(dir[0])
    return


def make_ouput_dir(output_dir:str, labes_dir_list)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lopDir0 in ("train", "val"):
        firstDir = os.path.join(output_dir, lopDir0)
        if not os.path.exists(firstDir):
            #shutil.rmtree(firstDir)
            os.makedirs(firstDir)
        for lopDir1 in labes_dir_list:
            secondDir = os.path.join(firstDir, lopDir1)
            if not os.path.exists(secondDir):
                os.makedirs(secondDir)
    return



def mat2_yolo(output_dir, pa100k_data)->None:
    labes_dir_list = []
    get_labels(pa100k_data, labes_dir_list)
    print(labes_dir_list)
    #df_data.to_csv("%s.txt" % key, index=False)
    make_ouput_dir(output_dir, labes_dir_list)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
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
    mat2_yolo(args.output_dir, pa100k_data)
    return


if __name__ == "__main__":
    main_func()