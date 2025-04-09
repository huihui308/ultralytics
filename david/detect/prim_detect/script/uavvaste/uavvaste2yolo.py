import json
import os
import shutil


def coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO JSON annotations to YOLO format.
    """
    # Load the COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Map image_id to file_name for quick lookup
    image_id_to_file = {image['id']: image['file_name'] for image in coco_data['images']}

    # Map category_id to a class index (YOLO needs classes as integers starting from 0)
    category_id_to_class = {category['id']: idx for idx, category in enumerate(coco_data['categories'])}

    # Dictionary to hold annotations per image
    image_annotations = {}

    # Process annotations
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # COCO bbox format: [x_min, y_min, width, height]

        # Get image dimensions
        image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
        img_width = image_info['width']
        img_height = image_info['height']

        # Normalize the bounding box to YOLO format
        x_min, y_min, box_width, box_height = bbox
        x_center = (x_min + box_width / 2) / img_width
        y_center = (y_min + box_height / 2) / img_height
        norm_width = box_width / img_width
        norm_height = box_height / img_height

        # Get the corresponding class index
        class_idx = category_id_to_class[category_id]

        # Prepare annotation in YOLO format
        yolo_annotation = f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

        # Append annotation to the list of annotations for this image
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(yolo_annotation)

    # Write annotations to .txt files
    for image_id, annotations in image_annotations.items():
        # Get the image file name and strip the extension to use as the base for the .txt file
        image_file_name = image_id_to_file[image_id]
        base_name = os.path.splitext(image_file_name)[0]
        txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

        # Write annotations to the file
        with open(txt_file_path, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')

    print(f"Annotations have been successfully converted and saved to {output_dir}")


def organize_data_using_split(split_json_path, images_dir, labels_dir, output_dir):
    """
    Organize images and labels into train, val, and test directories with nested subdirectories.
    Each split (train, val, test) will have its own 'images' and 'labels' directories.
    """
    # Load the split JSON file
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    # Ensure output directories exist for train, val, and test splits
    for split_type in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split_type, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_type, 'labels'), exist_ok=True)

    # Function to copy files
    def copy_files(file_list, split_type):
        for file_name in file_list:
            # Copy image to <split>/images/
            src_image_path = os.path.join(images_dir, file_name)
            dst_image_path = os.path.join(output_dir, split_type, 'images', file_name)
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
                # os.symlink(src_image_path, dst_image_path)
            else:
                print(f"Image file not found: {src_image_path}")

            # Copy corresponding label to <split>/labels/
            label_file_name = os.path.splitext(file_name)[0] + '.txt'
            src_label_path = os.path.join(labels_dir, label_file_name)
            dst_label_path = os.path.join(output_dir, split_type, 'labels', label_file_name)

            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
                # os.symlink(src_label_path, dst_label_path)
            else:
                print(f"Label file not found: {src_label_path}")

    # Copy train set
    print(f"Copying {len(split_data.get('train', []))} images to train/images and train/labels...")
    copy_files(split_data.get('train', []), 'train')

    # Copy val set
    print(f"Copying {len(split_data.get('val', []))} images to val/images and val/labels...")
    copy_files(split_data.get('val', []), 'val')

    # Copy test set (if it exists)
    if 'test' in split_data and split_data['test']:
        print(f"Copying {len(split_data['test'])} images to test/images and test/labels...")
        copy_files(split_data['test'], 'test')

    print("Data has been successfully organized into train, val, and test sets with nested directories.")


if __name__ == "__main__":
    # Paths for UAVVaste dataset
    coco_json_path = '/home/david/docker/share/dataset/uavvaste/annotations/annotations.json'              # Path to UAVVaste COCO JSON file
    split_json_path = '/home/david/docker/share/dataset/uavvaste/annotations/train_val_test_distribution_file.json'  # Path to split JSON file
    images_dir = '/home/david/docker/share/dataset/uavvaste/images'                           # Directory containing UAVVaste images
    yolo_labels_dir = '/home/david/docker/share/dataset/uavvaste/labels'                          # Output directory for YOLO labels
    output_dir = '/home/david/docker/share/dataset/uavvaste/yolo'                              # Final output directory for YOLO dataset

    # Step 1: Convert COCO annotations to YOLO format
    print("Converting COCO annotations to YOLO format...")
    coco_to_yolo(coco_json_path, yolo_labels_dir)

    # Step 2: Organize data into train, val, and test sets using the split file
    print("Organizing data into train, val, and test sets...")
    organize_data_using_split(split_json_path, images_dir, yolo_labels_dir, output_dir)