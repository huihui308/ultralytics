#
# cmd:
#       rm -rf tricycle_datasets;python3 convert_dataset.py --input_dir=/home/david/dataset/classification/tricycle/haitian_label_20230518 --output_dir=./tricycle_datasets
#
import scipy
import scipy.io
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path
from easydict import EasyDict
import os, sys, math, json, shutil, random, datetime, signal, argparse


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


def make_ouput_dir(output_dir:str)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for i in range(32):
            second_dir = os.path.join(first_dir, 'class' + str(i).zfill(4))
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def deal_files(files_list, output_dir)->None:
    for image_file in files_list:
        json_file = os.path.splitext(image_file)[0] + '.json'
        if not Path( json_file ).is_file():
            prRed('{} not exist, continue'.format(json_file))
            continue
        with open(json_file, "r") as fp:
            json_data = json.load(fp, encoding='utf-8')
            print(json_data['color']['name'])
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
    files_list = []
    for root, dirs, files in os.walk(args.input_dir):
        for one_file in files:
            file_name, file_type = os.path.splitext(one_file)
            #if file_type != '.json':
            if file_type not in ('.jpg', '.png', '.bmp'):
                continue
            files_list.append( os.path.join(root, one_file) )
    print(len(files_list), files_list[0])
    deal_files(files_list, args.output_dir)
    return


if __name__ == "__main__":
    main_func()