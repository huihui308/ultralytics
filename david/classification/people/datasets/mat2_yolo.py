#
# cmd:
#       python3 mat2_yolo.py.py --mat_file=/home/david/dataset/classification/people/annotation.mat
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
    return parser.parse_args(args)


def mat2txt(data, key):
    subdata = data[key]
    dfdata = pd.DataFrame(subdata)
    if key == 'attributes':
        print(dfdata)
    dfdata.to_csv("%s.txt" % key, index=False)


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    print(args.mat_file)
    pa100k_data = scipy.io.loadmat(args.mat_file)
    print(type(pa100k_data))
    #print(pa100k_data)
    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    print(type(train_image_name), type(val_image_name), type(test_image_name))
    train_label, val_label, test_label = pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']
    key_list = ["attributes", "test_images_name", "test_label",
                "train_images_name", "train_label",
                "val_images_name", "val_label"]
    for key in key_list:
        mat2txt(pa100k_data, key)

    return


if __name__ == "__main__":
    main_func()