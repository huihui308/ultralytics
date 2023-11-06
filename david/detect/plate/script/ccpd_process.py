# -*- coding: utf-8 -*-
#
# $ python3 ccpd_process.py --ccpd_dir=/home/david/dataset/lpd_lpr/detect_plate_datasets
#
#
import cv2
import numpy as np
from tqdm import tqdm
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
        "--ccpd_dir",
        type = str,
        required = True,
        help = "Labelme dir."
    )
    return parser.parse_args(args)


def all_file_path(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            all_file_path(os.path.join(rootPath,temp), allFIleList)
    return


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts=pts[:4,:]
    rect = np.zeros((5, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect


def get_partical_ccpd():
    ccpd_dir = r"/mnt/Gpan/BaiduNetdiskDownload/CCPD1/CCPD2020/ccpd_green"
    save_Path = r"ccpd/green_plate"
    folder_list = os.listdir(ccpd_dir)
    for folder_name in folder_list:
        count=0
        folder_path = os.path.join(ccpd_dir,folder_name)
        if os.path.isfile(folder_path):
            continue
        if folder_name == "ccpd_fn":
            continue
        name_list = os.listdir(folder_path)
        
        save_folder=save_Path
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for name in name_list:
            file_path = os.path.join(folder_path,name)
            count+=1
            if count>1000:
                break
            new_file_path =os.path.join(save_folder,name)
            shutil.move(file_path,new_file_path)
            print(count,new_file_path)


def get_rect_and_landmarks(img_path):
    file_name = img_path.split("/")[-1].split("-")
    #print(len(file_name))
    if len(file_name) != 7:
        return (None, None, None)
    landmarks_np = np.zeros((5, 2))
    rect = file_name[2].split("_")
    landmarks = file_name[3].split("_")
    rect_str = "&".join(rect)
    landmarks_str = "&".join(landmarks)
    rect = rect_str.split("&")
    landmarks = landmarks_str.split("&")
    rect = [int(x) for x in rect]
    landmarks = [int(x) for x in landmarks]
    for i in range(4):
        landmarks_np[i][0] = landmarks[2*i]
        landmarks_np[i][1] = landmarks[2*i+1]
    #    middle_landmark_w =int((landmarks[4]+landmarks[6])/2) 
    #    middle_landmark_h =int((landmarks[5]+landmarks[7])/2) 
    #    landmarks.append(middle_landmark_w)
    #    landmarks.append(middle_landmark_h)
    landmarks_np_new = order_points(landmarks_np)
    #    landmarks_np_new[4]=np.array([middle_landmark_w,middle_landmark_h])
    return (rect, landmarks, landmarks_np_new)


def x1x2y1y2_yolo(rect,landmarks,img):
    h,w,c =img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    annotation = np.zeros((1, 14))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks[0] / w  # l0_x
    annotation[0, 5] = landmarks[1] / h  # l0_y
    annotation[0, 6] = landmarks[2] / w  # l1_x
    annotation[0, 7] = landmarks[3] / h  # l1_y
    annotation[0, 8] = landmarks[4] / w  # l2_x
    annotation[0, 9] = landmarks[5] / h # l2_y
    annotation[0, 10] = landmarks[6] / w  # l3_x
    annotation[0, 11] = landmarks[7] / h  # l3_y
    # annotation[0, 12] = landmarks[8] / w  # l4_x
    # annotation[0, 13] = landmarks[9] / h  # l4_y
    return annotation


def xywh2yolo(rect,landmarks_sort,img):
    h,w,c =img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    annotation = np.zeros((1, 12))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    # annotation[0, 12] = landmarks_sort[4][0] / w  # l4_x
    # annotation[0, 13] = landmarks_sort[4][1] / h  # l4_y
    return annotation


def yolo2x1y1x2y2(annotation,img):
    h,w,c = img.shape
    rect= annotation[:,0:4].squeeze().tolist()
    landmarks=annotation[:,4:].squeeze().tolist()
    rect_w = w*rect[2]
    rect_h =h*rect[3]
    rect_x =int(rect[0]*w-rect_w/2)
    rect_y = int(rect[1]*h-rect_h/2)
    new_rect=[rect_x,rect_y,rect_x+rect_w,rect_y+rect_h]
    for i in range(5):
        landmarks[2*i]=landmarks[2*i]*w
        landmarks[2*i+1]=landmarks[2*i+1]*h
    return (new_rect, landmarks)


def write_lable(file_path):
    pass


def process_files(img_path)->None:
    text_path = img_path.replace(".jpg", ".txt")
    img = cv2.imread(img_path)
    rect, landmarks, landmarks_sort = get_rect_and_landmarks(img_path)
    if rect is None:
        #prRed('{} not CCPD format, continue'.format(img_path))
        return
    # annotation=x1x2y1y2_yolo(rect,landmarks,img)
    annotation = xywh2yolo(rect, landmarks_sort,img)
    str_label = "0 "
    for i in range(len(annotation[0])):
        str_label = str_label + " " + str(annotation[0][i])
    str_label = str_label.replace('[', '').replace(']', '')
    str_label = str_label.replace(',', '') + '\n'
    with open(text_path,"w") as f:
        f.write(str_label)
    #print(img_path, str_label)
    return


def main_func(args = None)->None:
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    prYellow('CCPD dir:{}'.format(args.ccpd_dir))
    file_list = []
    all_file_path(args.ccpd_dir, file_list)
    #print(len(file_list), file_list[0])
    pbar = enumerate(file_list)
    pbar = tqdm(pbar, total=len(file_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (i, img_path) in pbar:
        # img_path = r"ccpd_yolo_test/02-90_85-173&466_452&541-452&553_176&556_178&463_454&460-0_0_6_26_15_26_32-68-53.jpg"
        process_files(img_path)
    return


if __name__ == '__main__':
    main_func()