import json

def get_class_list(coco_json_path):
    """
    Extract the list of detectable object types (classes) from the COCO JSON file.
    """
    # Load the COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract the list of categories
    categories = coco_data.get('categories', [])
    class_list = [category['name'] for category in categories]

    # Print the number of classes
    print(f"Number of Classes: {len(class_list)}")

    # Print the list of classes
    print("List of Classes:")
    for idx, class_name in enumerate(class_list, start=1):
        print(f"{idx}. {class_name}")

    return class_list


if __name__ == "__main__":
    # Path to the UAVVaste COCO JSON file
    coco_json_path = '/home/david/docker/share/dataset/uavvaste/annotations/annotations.json'

    # Get the list of detectable object types
    class_list = get_class_list(coco_json_path)