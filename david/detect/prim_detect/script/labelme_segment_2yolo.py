import os
import json
import shutil
import uuid  # For generating unique random names
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def polygon_to_bbox(polygon: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Convert a polygon (list of points) to a bounding box.
    :param polygon: List of [x, y] points defining the polygon.
    :return: Bounding box as (x_min, y_min, x_max, y_max).
    """
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def normalize_bbox(bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates to YOLO format.
    :param bbox: Bounding box as (x_min, y_min, x_max, y_max).
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: Normalized bounding box as (x_center, y_center, width, height).
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def create_yolo_dataset(labelme_dir: str, output_dir: str, class_mapping: dict, train_ratio: float = 0.8):
    """
    Convert a LabelMe dataset to YOLO format with random naming and split into train/val directories.
    :param labelme_dir: Directory containing LabelMe JSON files and images.
    :param output_dir: Directory to save the YOLO dataset.
    :param class_mapping: Dictionary mapping class names to class IDs.
    :param train_ratio: Ratio of training data (default is 0.8).
    """
    # Create output directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Collect all JSON files
    json_files = [f for f in os.listdir(labelme_dir) if f.endswith(".json")]
    train_files, val_files = train_test_split(json_files, train_size=train_ratio, random_state=42)

    # Process each file
    for json_file in json_files:
        json_path = os.path.join(labelme_dir, json_file)

        # Generate a random name for the image and label
        random_name = str(uuid.uuid4())  # Generates a unique random UUID
        image_ext = ".jpg"  # Default image extension; adjust if needed

        # Determine the target directories based on the split
        if json_file in train_files:
            target_images_dir = train_images_dir
            target_labels_dir = train_labels_dir
        else:
            target_images_dir = val_images_dir
            target_labels_dir = val_labels_dir

        # Load the LabelMe JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract image dimensions and image file name
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        img_file = data['imagePath']
        img_src_path = os.path.join(labelme_dir, img_file)
        img_dst_path = os.path.join(target_images_dir, f"{random_name}{image_ext}")

        # Copy the image to the target directory with the new random name
        shutil.copy(img_src_path, img_dst_path)

        # Prepare the output label file path
        label_file = f"{random_name}.txt"
        label_dst_path = os.path.join(target_labels_dir, label_file)

        # Write YOLO annotations
        with open(label_dst_path, 'w') as f:
            for shape in data['shapes']:
                class_name = shape['label']
                polygon = shape['points']

                # Convert polygon to bounding box
                bbox = polygon_to_bbox(polygon)

                # Normalize bounding box coordinates
                normalized_bbox = normalize_bbox(bbox, img_width, img_height)

                # Get class ID from the mapping
                if class_name not in class_mapping:
                    raise ValueError(f"Class '{class_name}' not found in class mapping.")
                class_id = class_mapping[class_name]

                # Write to the YOLO file
                f.write(f"{class_id} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]}\n")

    print("Dataset conversion and split completed!")

# Example usage
if __name__ == "__main__":
    # Define the class mapping (e.g., {"cat": 0, "dog": 1})
    class_mapping = {
        "glass": 0,
        "foam": 1,
        "beer": 2,
        "tap": 3,
    }

    # Paths
    # light_shoot2  normal  side_cup_much_foam  straight_cup_add_beer
    labelme_dir = "/home/david/docker/share/dataset/beer/straight_cup_add_beer"
    output_dir = "/home/david/docker/share/dataset/beer/yolo"

    # Create the YOLO dataset
    create_yolo_dataset(labelme_dir, output_dir, class_mapping)