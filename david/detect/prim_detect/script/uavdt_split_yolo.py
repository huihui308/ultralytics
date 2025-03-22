#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    $ python uavdt_split_yolo.py

    https://github.com/forever208/yolov5_train_on_UAVDT
"""

from tqdm import tqdm
import os, sys, shutil, random, datetime





def split_data(data_list, out_images_dir, out_labels_dir)->None:
    for image_file_path in data_list:
        # Split the path into components
        parts = image_file_path.split(os.sep)

        # Replace 'train' with 'val'
        parts[-2] = 'labels'

        label_name = parts[-1]
        base_name, extension = os.path.splitext(label_name)
        # Create the new file name with .txt extension
        label_name = f"{base_name}.txt"
        parts[-1] = label_name

        # Reconstruct the path
        label_file_path = os.sep.join(parts)
        # print(image_file_path, label_file_path)

        image_name = os.path.basename(image_file_path)
        label_name = os.path.basename(label_file_path)
        # Split the file name into base and extension
        base_name, extension = os.path.splitext(label_name)
        # Create the new file name with .txt extension
        label_name = f"{base_name}.txt"
        # print(image_name, label_name)

        out_image_path = os.path.join(out_images_dir, image_name)
        out_label_path = os.path.join(out_labels_dir, label_name)
        os.symlink(image_file_path, out_image_path)
        os.symlink(label_file_path, out_label_path)
    return


def deal_files(old_dir, output_dir)->None:
    if not os.path.exists(old_dir):
        sys.stdout.write('\r>> {}: Dir: {} not exist, return\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), old_dir))
        return

    images_dir_list = []
    labels_dir_list = []
    images_dir_list.append( os.path.join(old_dir, 'train', 'images') )
    images_dir_list.append( os.path.join(old_dir, 'val', 'images') )
    labels_dir_list.append( os.path.join(old_dir, 'train', 'labels') )
    labels_dir_list.append( os.path.join(old_dir, 'val', 'labels') )
    # print(images_dir_list, labels_dir_list)
    for tmp_dir in images_dir_list:
        if not os.path.exists(tmp_dir):
            sys.stdout.write('\r>> {}: Dir: {} not exist, return\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tmp_dir))
            return
    for tmp_dir in labels_dir_list:
        if not os.path.exists(tmp_dir):
            sys.stdout.write('\r>> {}: Dir: {} not exist, return\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tmp_dir))
            return

    images_list = []
    for tmp_dir in images_dir_list:
        # Iterate over the files in the directory
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)  # Get the full file path            
            # Check if it's a file (not a subdirectory)
            if not os.path.isfile(file_path):
                continue

            # Split the file name into base and extension
            base_name, extension = os.path.splitext(filename)

            # Create the new file name with .txt extension
            new_file_name = f"{base_name}.txt"
            new_file_name = os.path.join(old_dir, 'train', 'labels', new_file_name)
            if not os.path.exists(new_file_name):
                # sys.stdout.write('\r>> {}: Dir: {} not exist, return\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), new_file_name))
                # print(filename, new_file_name)
                continue
            images_list.append(file_path)
    sys.stdout.write('\r>> {}: Images size: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(images_list)))
    random.shuffle(images_list)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_train_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir)
    out_train_images_dir = os.path.join(output_dir, 'train', 'images')
    if not os.path.exists(out_train_images_dir):
        os.makedirs(out_train_images_dir)
    out_train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    if not os.path.exists(out_train_labels_dir):
        os.makedirs(out_train_labels_dir)

    out_val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(out_val_dir):
        os.makedirs(out_val_dir)
    out_val_images_dir = os.path.join(output_dir, 'val', 'images')
    if not os.path.exists(out_val_images_dir):
        os.makedirs(out_val_images_dir)
    out_val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    if not os.path.exists(out_val_labels_dir):
        os.makedirs(out_val_labels_dir)

    split_index = int(len(images_list) * 0.8)
    train_list = images_list[:split_index]
    val_list = images_list[split_index:]
    sys.stdout.write('\r>> {}: Train size: {}, val size: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(train_list), len(val_list)))

    split_data(train_list, out_train_images_dir, out_train_labels_dir)
    split_data(val_list, out_val_images_dir, out_val_labels_dir)

    sys.stdout.write('\r>> {}: Convert success\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return


if __name__ == "__main__":
    old_dir = '/home/david/dataset/drone/UAVDTOrigin/consumer/UAVDT'
    output_dir = '/home/david/dataset/drone/uavdt-split'
    deal_files(old_dir, output_dir)