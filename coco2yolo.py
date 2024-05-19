import json
import os
import argparse
import logging
from pathlib import Path

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_coco_to_yolo(size, box):
    """
    Convert COCO bounding box format to YOLO format.
    size: (width, height) of the image
    box: [x_top_left, y_top_left, width, height] COCO bbox
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def process_annotations(dataset_path):
    """
    Process COCO annotations and convert them to YOLO format.
    """
    for split in ['train', 'val', 'test']:
        label_path = Path(dataset_path) / "labels" / split / 'coco.json'
        if not label_path.exists():
            logger.warning(f"File not found: {label_path}")
            continue

        with open(label_path) as f:
            data = json.load(f)
        
        # Mapping from image id to filename
        image_info = {img['id']: img for img in data['images']}
        
        # Process each annotation
        for ann in data['annotations']:
            img_id = ann['image_id']
            coco_bbox = ann['bbox']
            category_id = ann['category_id'] - 1  # Assuming category IDs are 1-indexed in COCO
            img_filename = Path(image_info[img_id]['file_name'])
            img_size = (image_info[img_id]['width'], image_info[img_id]['height'])
            yolo_bbox = convert_coco_to_yolo(img_size, coco_bbox)
            
            txt_path = label_path.parent / (img_filename.stem + '.txt')
            with open(txt_path, 'a') as file:
                file.write(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")
            logger.info(f"Processed annotation for image: {img_filename}")

def create_yaml_file(dataset_path):
    """
    Create a .yml file for the dataset configuration using class names extracted from coco.json.
    """
    label_path = dataset_path / "labels" / 'train' / 'coco.json'
    if not label_path.exists():
        logger.error(f"File not found: {label_path}")
        return

    with open(label_path) as f:
        data = json.load(f)
    class_names = {category['id'] - 1: category['name'] for category in data['categories']}

    # Ensure the classes are sorted by their ids and formatted correctly
    sorted_class_names = sorted(class_names.items())
    class_entries = "\n".join([f"  {id}: {name}" for id, name in sorted_class_names])

    yaml_content = f"""path: {dataset_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
{class_entries}
    """

    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as file:
        file.write(yaml_content.strip())
    logger.info(f"YAML file created at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COCO annotations and create YOLO dataset.")
    parser.add_argument("dataset_path", help="Path to the root directory of the dataset.")
    
    args = parser.parse_args()
    dataset_root = Path(args.dataset_path)
    
    process_annotations(dataset_root)
    create_yaml_file(dataset_root)
