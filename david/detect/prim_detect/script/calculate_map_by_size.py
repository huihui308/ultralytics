from ultralytics import YOLO
import cv2
import numpy as np
import os
from sklearn.metrics import auc


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) given recall and precision curves.
    """
    # Append sentinel values to beginning and end
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Identify points where recall changes
    indices = np.where(np.diff(recall))[0]

    # Calculate AP as the area under the precision-recall curve
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    """
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Compute the area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def parse_yolo_annotation(annotation_path, image_width, image_height):
    """
    Parse a YOLO annotation file and convert normalized coordinates to absolute pixel values.
    
    Args:
        annotation_path: Path to the YOLO annotation file.
        image_width: Width of the image.
        image_height: Height of the image.
    
    Returns:
        boxes: List of bounding boxes in [x1, y1, x2, y2] format.
    """
    boxes = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, data)
            
            # Convert normalized coordinates to absolute values
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2])
    return boxes


def categorize_boxes_by_size(boxes, size_thresholds):
    """
    Categorize bounding boxes into small, medium, and large based on their area.
    
    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...].
        size_thresholds: Tuple (small_max, medium_max) defining size categories.
                         Boxes with area <= small_max are "small",
                         Boxes with area > small_max and <= medium_max are "medium",
                         Boxes with area > medium_max are "large".
    
    Returns:
        categorized_boxes: Dictionary {'small': [...], 'medium': [...], 'large': [...]}.
    """
    small_max, medium_max = size_thresholds
    categorized_boxes = {'small': [], 'medium': [], 'large': []}

    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height

        if area <= small_max:
            categorized_boxes['small'].append(box)
        elif area <= medium_max:
            categorized_boxes['medium'].append(box)
        else:
            categorized_boxes['large'].append(box)

    return categorized_boxes


def calculate_map(predictions, ground_truths, size_thresholds, iou_threshold=0.5):
    """
    Calculate mAP for bounding boxes categorized by size (small, medium, large).
    
    Args:
        predictions: List of predicted bounding boxes and scores.
                     Format: [{'boxes': [[x1, y1, x2, y2], ...], 'scores': [score1, score2, ...]}, ...]
        ground_truths: List of ground truth bounding boxes.
                       Format: [[x1, y1, x2, y2], ...]
        size_thresholds: Tuple (small_max, medium_max) defining size categories.
        iou_threshold: Threshold for Intersection over Union (IoU).
    
    Returns:
        mAP_by_size: Dictionary {'small': mAP_small, 'medium': mAP_medium, 'large': mAP_large}.
    """
    # Categorize ground truth boxes by size
    gt_by_size = categorize_boxes_by_size(ground_truths, size_thresholds)

    # Initialize results
    mAP_by_size = {}

    # Calculate mAP for each size category
    for size_category, gt_boxes in gt_by_size.items():
        true_positives = []
        false_positives = []
        scores = []

        # Filter predictions by size category
        pred_boxes = []
        for pred in predictions:
            for box in pred['boxes']:
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height

                if size_category == 'small' and area <= size_thresholds[0]:
                    pred_boxes.append(box)
                elif size_category == 'medium' and size_thresholds[0] < area <= size_thresholds[1]:
                    pred_boxes.append(box)
                elif size_category == 'large' and area > size_thresholds[1]:
                    pred_boxes.append(box)

        # Match predictions to ground truths
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                true_positives.append(1)
                false_positives.append(0)
                # Remove matched ground truth to avoid double counting
                gt_boxes.pop(best_gt_idx)
            else:
                true_positives.append(0)
                false_positives.append(1)

            scores.append(1.0)  # Placeholder for confidence scores (replace with actual scores)

        # Sort by confidence scores
        sorted_indices = np.argsort(-np.array(scores))
        true_positives = np.array(true_positives)[sorted_indices]
        false_positives = np.array(false_positives)[sorted_indices]

        # Compute cumulative TP and FP
        cum_true_positives = np.cumsum(true_positives)
        cum_false_positives = np.cumsum(false_positives)

        # Compute precision and recall
        precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
        recall = cum_true_positives / (len(gt_boxes) + 1e-10)

        # Compute AP for this size category
        ap = compute_ap(recall, precision)
        mAP_by_size[size_category] = ap

    return mAP_by_size


# Load the YOLOv11 model
model = YOLO("runs/detect/yolo12x-4heads-900epoches-visdrone-20250310/weights/best.pt")  # Replace with your model path

# Path to the validation directory
validation_dir = "/home/david/dataset/drone/VisDroneOrigin/VisDrone2019-DET-val"
images_dir = os.path.join(validation_dir, "images")
labels_dir = os.path.join(validation_dir, "labels")

# Define size thresholds (small <= 4096, medium <= 16384, large > 16384)
size_thresholds = (1024, 9216)

# Initialize metrics
total_mAP_by_size = {'small': [], 'medium': [], 'large': []}

# Iterate through all images in the validation directory
for image_filename in os.listdir(images_dir):
    if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    # Load the image
    image_path = os.path.join(images_dir, image_filename)
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Resize the image to 640x640
    resized_image = cv2.resize(image, (640, 640))

    # Perform inference on the resized image
    results = model(resized_image)

    # Extract predictions
    predictions = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        predictions.append({'boxes': boxes, 'scores': scores})

    # Parse ground truth annotations
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_filename)
    if not os.path.exists(label_path):
        print(f"Label file not found for {image_filename}. Skipping...")
        continue

    ground_truths = parse_yolo_annotation(label_path, original_width, original_height)

    # Calculate mAP by size
    mAP_by_size = calculate_map(predictions, ground_truths, size_thresholds)

    # Accumulate results
    for size_category, ap in mAP_by_size.items():
        total_mAP_by_size[size_category].append(ap)

# Compute mean mAP across all images
mean_mAP_by_size = {size_category: np.mean(ap_list) for size_category, ap_list in total_mAP_by_size.items()}
print(f"Mean mAP by size: {mean_mAP_by_size}")