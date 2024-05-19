import json
import shutil
import random
import argparse
import logging
import logging.config

from pathlib import Path
from utils.json import get_json_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_images(images, src_dir, dest_dir):
    """
    Copy selected images to a specified directory.
    """
    for image in images:
        try:
            src_path = Path(src_dir) / image['file_name']
            dest_path = Path(dest_dir) / src_path.name
            shutil.copy(src_path, dest_path)
            logger.info(f"Successfully copied {src_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dest_path}: {e}")

def split_dataset(images_dir, labels_json_path, output_dir, train_ratio=0.75, val_ratio=0.1):
    """
    Splits a COCO dataset into training, validation, and testing sets based on given ratios.
    """
    try:
        logger.info("Loading COCO annotations...")
        coco_data = get_json_from_file(labels_json_path)
    except FileNotFoundError:
        logger.error(f"File not found: {labels_json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {labels_json_path}")
        return

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return

    # Extract image and annotation details
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    # Validate ratios
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio <= 1):
        logger.error("Invalid training/validation ratios.")
        return

    random.shuffle(images)
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    def filter_annotations(images_set):
        image_ids = {image['id'] for image in images_set}
        return [annotation for annotation in annotations if annotation['image_id'] in image_ids]

    def create_coco_subset(images, annotations):
        return {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }

    for type, images_set in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        try:
            images_output_path = output_dir / "images" / type
            images_output_path.mkdir(parents=True, exist_ok=True)

            labels_output_path = output_dir / "labels" / type
            labels_output_path.mkdir(parents=True, exist_ok=True)

            copy_images(images_set, images_dir, images_output_path)

            coco_file = create_coco_subset(images_set, filter_annotations(images_set))
            with open(labels_output_path / "coco.json", 'w') as file:
                json.dump(coco_file, file, indent=4)
            logger.info(f"Dataset for {type} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to process data for {type}: {e}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into training, validation, and testing sets.")
    parser.add_argument("images_dir", help="Path to the input directory containing images.")
    parser.add_argument("coco_json_path", help="Path to the COCO JSON file containing annotations.")
    parser.add_argument("output_dir", help="Path to the root output directory for training, validation, and testing sets.")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proportion of images for training (default: 0.75)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of images for validation (default: 0.1)")

    args = parser.parse_args()
    split_dataset(args.images_dir, args.coco_json_path, args.output_dir, args.train_ratio, args.val_ratio)
