#!/usr/bin/env python3
"""
XML to YOLO Format Converter
Supports RDD2022 and MWPD datasets
Based on RoadDamageDetection repository implementation
"""

import os
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XMLToYOLOConverter:
    """Convert Pascal VOC XML annotations to YOLO format"""

    def __init__(self, dataset_type: str = "auto"):
        # RDD2022 class mapping
        self.rdd_class_mapping = {
            "D00": 0,  # Longitudinal Crack
            "D10": 1,  # Transverse Crack
            "D20": 2,  # Alligator Crack
            "D40": 3,  # Potholes
            # Additional classes (not used in filtered dataset)
            "D01": 4,
            "D11": 5,
            "D43": 6,
            "D44": 7,
            "D50": 8,
        }

        self.rdd_class_names = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]

        # MWPD class mapping (single class: pothole)
        self.mwpd_class_mapping = {
            "pothole": 0,
            "Pothole": 0,
        }

        self.mwpd_class_names = ["Pothole"]

        # Combined mapping for rdd_mwpd.yaml
        self.combined_class_names = [
            "Longitudinal Crack",  # 0 - RDD only
            "Transverse Crack",    # 1 - RDD only
            "Alligator Crack",     # 2 - RDD only
            "Potholes",            # 3 - Combined RDD + MWPD potholes
        ]

        self.dataset_type = dataset_type

    def detect_dataset_type(self, dataset_root: str) -> str:
        """Auto-detect dataset type"""
        dataset_path = Path(dataset_root)
        
        # Check for RDD2022 structure
        if (dataset_path / "RDD2022_all_countries").exists():
            return "rdd2022"
        
        # Check for MWPD structure (train/test/valid folders with images and labels)
        has_train = (dataset_path / "train").exists()
        has_images = (dataset_path / "train" / "images").exists() if has_train else False
        has_labels = (dataset_path / "train" / "labels").exists() if has_train else False
        
        if has_train and has_images and has_labels:
            return "mwpd"
        
        logger.warning(f"Could not detect dataset type in {dataset_root}")
        return "unknown"

    def convert_pascal_to_yolo(self, xml_file_path: str, dataset_type: str = "rdd2022") -> bool:
        """Convert single Pascal VOC XML file to YOLO format"""

        try:
            # Parse XML
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Get image dimensions
            size = root.find("size")
            if size is None:
                logger.warning(f"No size information in {xml_file_path}")
                return False

            image_width = int(size.find("width").text)
            image_height = int(size.find("height").text)

            # Process bounding boxes
            bounding_boxes = []
            class_ids = []

            for obj in root.findall("object"):
                # Get class name
                class_name = obj.find("name").text

                if dataset_type == "rdd2022":
                    # Map class to ID (filter out unwanted classes)
                    class_id = self.rdd_class_mapping.get(class_name, 10)
                    # Only keep the main 4 damage types
                    if class_id >= 4:
                        continue
                else:  # mwpd
                    class_id = self.mwpd_class_mapping.get(class_name, -1)
                    if class_id == -1:
                        continue

                class_ids.append(class_id)

                # Get bounding box
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Convert to YOLO format (normalized center coordinates)
                width = xmax - xmin
                height = ymax - ymin
                center_x = xmin + (width / 2)
                center_y = ymin + (height / 2)

                # Normalize
                norm_center_x = round(center_x / image_width, 6)
                norm_center_y = round(center_y / image_height, 6)
                norm_width = round(width / image_width, 6)
                norm_height = round(height / image_height, 6)

                bounding_boxes.append(
                    [norm_center_x, norm_center_y, norm_width, norm_height]
                )

            # Create output file path
            xml_path = Path(xml_file_path)
            output_dir = xml_path.parents[2] / "labels"
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"{xml_path.stem}.txt"

            # Write YOLO format annotations
            with open(output_file, "w") as f:
                for i, bbox in enumerate(bounding_boxes):
                    class_id = class_ids[i]
                    annotation = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                    f.write(annotation)

            return True

        except Exception as e:
            logger.error(f"Error converting {xml_file_path}: {e}")
            return False

    def convert_rdd_dataset(self, dataset_root: str, countries: List[str] = None) -> bool:
        """Convert RDD2022 dataset from XML to YOLO format"""

        if countries is None:
            countries = ["Japan", "India"]

        dataset_path = Path(dataset_root)

        for country in countries:
            country_dir = (
                dataset_path
                / "RDD2022_all_countries"
                / country
                / "train"
                / "annotations"
                / "xmls"
            )

            if not country_dir.exists():
                logger.warning(f"Country directory not found: {country_dir}")
                continue

            logger.info(f"Converting annotations for {country}")

            # Get all XML files
            xml_files = list(country_dir.glob("*.xml"))

            # Convert each XML file
            success_count = 0
            for xml_file in tqdm(xml_files, desc=f"Converting {country}"):
                if self.convert_pascal_to_yolo(str(xml_file), "rdd2022"):
                    success_count += 1

            logger.info(
                f"Converted {success_count}/{len(xml_files)} files for {country}"
            )

        return True

    def convert_mwpd_dataset(self, dataset_root: str) -> bool:
        """Convert MWPD dataset (already in YOLO format, just verify structure)"""
        
        dataset_path = Path(dataset_root)
        logger.info(f"Processing MWPD dataset at {dataset_root}")
        
        # MWPD dataset is already in YOLO format
        # Just verify the structure exists
        for split in ["train", "test", "valid"]:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                img_count = len(list(images_dir.glob("*")))
                lbl_count = len(list(labels_dir.glob("*.txt")))
                logger.info(f"MWPD {split}: {img_count} images, {lbl_count} labels")
            else:
                logger.warning(f"MWPD {split} directory not found or incomplete")
        
        return True

    def split_dataset(
        self,
        base_dir: str,
        country: str,
        output_dir: str,
        train_ratio: float = 0.9,
        background_percentage: float = 0.1,
    ) -> bool:
        """Split RDD2022 dataset into train/val and filter background images"""

        base_path = Path(base_dir) / "RDD2022_all_countries" / country / "train"
        output_path = Path(output_dir) / "rdd2022" / country

        # Get image and label lists
        images_dir = base_path / "images"
        labels_dir = base_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            logger.error(f"Images or labels directory not found for {country}")
            return False

        image_files = sorted(list(images_dir.glob("*")))
        label_files = sorted(list(labels_dir.glob("*")))

        logger.info(
            f"Found {len(image_files)} images and {len(label_files)} labels for {country}"
        )

        # Filter images: keep all with annotations + some background images
        filtered_images = []
        filtered_labels = []
        background_count = 0
        max_background = int(len(image_files) * background_percentage)

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"

            if label_file.exists():
                # Check if label file has content (not background)
                with open(label_file, "r") as f:
                    content = f.read().strip()

                if content:  # Has annotations
                    filtered_images.append(img_file)
                    filtered_labels.append(label_file)
                elif background_count < max_background:  # Background image
                    filtered_images.append(img_file)
                    filtered_labels.append(label_file)
                    background_count += 1

        logger.info(
            f"Filtered to {len(filtered_images)} images (including {background_count} background)"
        )

        # Split into train/val
        dataset_length = len(filtered_images)
        split_point = int(train_ratio * dataset_length)

        # Shuffle with seed for reproducibility
        random.seed(1337)
        indices = list(range(dataset_length))
        random.shuffle(indices)

        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

        # Copy files to train/val directories
        splits = [("train", train_indices), ("val", val_indices)]

        for split_name, split_indices in splits:
            logger.info(f"Copying {split_name} files for {country}")

            # Create directories
            img_out_dir = output_path / "images" / split_name
            lbl_out_dir = output_path / "labels" / split_name
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for idx in tqdm(split_indices, desc=f"Copying {split_name}"):
                # Copy image
                src_img = filtered_images[idx]
                dst_img = img_out_dir / src_img.name
                shutil.copy2(src_img, dst_img)

                # Copy label
                src_lbl = filtered_labels[idx]
                dst_lbl = lbl_out_dir / src_lbl.name
                shutil.copy2(src_lbl, dst_lbl)

        return True

    def prepare_mwpd_split(self, dataset_root: str, output_dir: str) -> bool:
        """Prepare MWPD dataset split (copy to output directory with class remapping)"""
        
        dataset_path = Path(dataset_root)
        output_path = Path(output_dir) / "mwpd"
        
        # MWPD uses train/test/valid, we'll map: train->train, valid->val, skip test
        split_mapping = {
            "train": "train",
            "valid": "val",
        }
        
        for src_split, dst_split in split_mapping.items():
            src_img_dir = dataset_path / src_split / "images"
            src_lbl_dir = dataset_path / src_split / "labels"
            
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                logger.warning(f"MWPD {src_split} directory not found")
                continue
            
            dst_img_dir = output_path / "images" / dst_split
            dst_lbl_dir = output_path / "labels" / dst_split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            images = list(src_img_dir.glob("*"))
            logger.info(f"Copying {len(images)} MWPD {src_split} images")
            
            for img_file in tqdm(images, desc=f"MWPD {src_split}"):
                # Copy image with mwpd prefix to avoid conflicts
                dst_img = dst_img_dir / f"mwpd_{img_file.name}"
                shutil.copy2(img_file, dst_img)
                
                # Copy label with mwpd prefix
                lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    dst_lbl = dst_lbl_dir / f"mwpd_{img_file.stem}.txt"
                    shutil.copy2(lbl_file, dst_lbl)
        
        return True

    def create_dataset_yaml(self, output_dir: str, countries: List[str], dataset_name: str = "rdd") -> str:
        """Create dataset.yaml file for YOLO training"""

        output_path = Path(output_dir)
        
        if dataset_name == "rdd":
            class_names = self.rdd_class_names
            yaml_file = output_path / "rdd_dataset.yaml"
            dataset_subdir = "rdd2022"
        else:  # mwpd
            class_names = self.mwpd_class_names
            yaml_file = output_path / "mwpd_dataset.yaml"
            dataset_subdir = "mwpd"

        # Create combined dataset structure
        yaml_content = {
            "path": str(output_path.absolute()),
            "train": f"{dataset_subdir}/train/images",
            "val": f"{dataset_subdir}/val/images",
            "nc": len(class_names),
            "names": class_names,
        }

        # Create combined train/val directories
        for split in ["train", "val"]:
            combined_img_dir = output_path / dataset_subdir / split / "images"
            combined_lbl_dir = output_path / dataset_subdir / split / "labels"
            combined_img_dir.mkdir(parents=True, exist_ok=True)
            combined_lbl_dir.mkdir(parents=True, exist_ok=True)

            if dataset_name == "rdd":
                # Combine all countries into single train/val
                for country in countries:
                    country_img_dir = output_path / dataset_subdir / country / "images" / split
                    country_lbl_dir = output_path / dataset_subdir / country / "labels" / split

                    if country_img_dir.exists():
                        # Copy images with country prefix to avoid conflicts
                        for img_file in country_img_dir.glob("*"):
                            dst_name = f"{country}_{img_file.name}"
                            shutil.copy2(img_file, combined_img_dir / dst_name)

                        # Copy labels with country prefix
                        for lbl_file in country_lbl_dir.glob("*"):
                            dst_name = f"{country}_{lbl_file.name}"
                            shutil.copy2(lbl_file, combined_lbl_dir / dst_name)

        # Write YAML manually to avoid dependency
        with open(yaml_file, "w") as f:
            f.write(f"path: {yaml_content['path']}\n")
            f.write(f"train: {yaml_content['train']}\n")
            f.write(f"val: {yaml_content['val']}\n")
            f.write(f"nc: {yaml_content['nc']}\n")
            f.write("names:\n")
            for name in yaml_content["names"]:
                f.write(f"  - {name}\n")

        logger.info(f"Dataset YAML created: {yaml_file}")
        return str(yaml_file)

    def create_combined_yaml(self, output_dir: str) -> str:
        """Create combined rdd_mwpd.yaml for training on both datasets"""
        
        output_path = Path(output_dir)
        yaml_file = output_path / "rdd_mwpd.yaml"
        
        # Combine RDD2022 and MWPD datasets
        for split in ["train", "val"]:
            combined_img_dir = output_path / split / "images"
            combined_lbl_dir = output_path / split / "labels"
            combined_img_dir.mkdir(parents=True, exist_ok=True)
            combined_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy RDD2022 files
            rdd_img_dir = output_path / "rdd2022" / split / "images"
            rdd_lbl_dir = output_path / "rdd2022" / split / "labels"
            
            if rdd_img_dir.exists():
                for img_file in rdd_img_dir.glob("*"):
                    shutil.copy2(img_file, combined_img_dir / f"rdd_{img_file.name}")
                
                for lbl_file in rdd_lbl_dir.glob("*.txt"):
                    shutil.copy2(lbl_file, combined_lbl_dir / f"rdd_{lbl_file.name}")
            
            # Copy MWPD files with class remapping (0 -> 3)
            mwpd_img_dir = output_path / "mwpd" / "images" / split
            mwpd_lbl_dir = output_path / "mwpd" / "labels" / split
            
            if mwpd_img_dir.exists():
                for img_file in mwpd_img_dir.glob("*"):
                    shutil.copy2(img_file, combined_img_dir / img_file.name)
                
                # Remap MWPD class from 0 to 3 (Potholes)
                for lbl_file in mwpd_lbl_dir.glob("*.txt"):
                    with open(lbl_file, "r") as f:
                        lines = f.readlines()
                    
                    with open(combined_lbl_dir / lbl_file.name, "w") as f:
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                # Change class from 0 to 3 (combine with RDD potholes)
                                parts[0] = "3"
                                f.write(" ".join(parts) + "\n")
        
        # Write combined YAML
        with open(yaml_file, "w") as f:
            f.write(f"path: {output_path.absolute()}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write(f"nc: {len(self.combined_class_names)}\n")
            f.write("names:\n")
            for name in self.combined_class_names:
                f.write(f"  - {name}\n")
        
        logger.info(f"Combined dataset YAML created: {yaml_file}")
        return str(yaml_file)


def main():
    """Main function for dataset conversion"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert RDD2022 and MWPD datasets to YOLO format"
    )
    parser.add_argument(
        "--dataset-root", required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--mwpd-root", help="Root directory of MWPD dataset (optional)"
    )
    parser.add_argument(
        "--output-dir", default="yolo_dataset", help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["Japan", "India"],
        help="Countries to process (for RDD2022)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9, help="Training split ratio"
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=0.1,
        help="Percentage of background images to keep",
    )
    parser.add_argument(
        "--create-combined",
        action="store_true",
        help="Create combined rdd_mwpd.yaml dataset",
    )

    args = parser.parse_args()

    converter = XMLToYOLOConverter()

    # Detect dataset type
    dataset_type = converter.detect_dataset_type(args.dataset_root)
    logger.info(f"Detected dataset type: {dataset_type}")

    if dataset_type == "rdd2022":
        # Process RDD2022 dataset
        logger.info("Converting RDD2022 XML annotations to YOLO format...")
        converter.convert_rdd_dataset(args.dataset_root, args.countries)

        logger.info("Splitting RDD2022 dataset and filtering background images...")
        for country in args.countries:
            converter.split_dataset(
                args.dataset_root,
                country,
                args.output_dir,
                args.train_ratio,
                args.background_ratio,
            )

        logger.info("Creating RDD2022 dataset configuration...")
        yaml_file = converter.create_dataset_yaml(args.output_dir, args.countries, "rdd")

    elif dataset_type == "mwpd":
        # Process MWPD dataset
        logger.info("Processing MWPD dataset...")
        converter.convert_mwpd_dataset(args.dataset_root)
        converter.prepare_mwpd_split(args.dataset_root, args.output_dir)
        
        logger.info("Creating MWPD dataset configuration...")
        yaml_file = converter.create_dataset_yaml(args.output_dir, [], "mwpd")

    # Process MWPD if separate root provided
    if args.mwpd_root:
        logger.info(f"Processing additional MWPD dataset from {args.mwpd_root}...")
        converter.convert_mwpd_dataset(args.mwpd_root)
        converter.prepare_mwpd_split(args.mwpd_root, args.output_dir)
        
        if dataset_type != "mwpd":
            logger.info("Creating MWPD dataset configuration...")
            converter.create_dataset_yaml(args.output_dir, [], "mwpd")

    # Create combined dataset if requested
    if args.create_combined:
        logger.info("Creating combined RDD2022+MWPD dataset...")
        combined_yaml = converter.create_combined_yaml(args.output_dir)
        print(f"\nCombined dataset config: {combined_yaml}")

    print("\n" + "=" * 50)
    print("DATASET CONVERSION COMPLETE")
    print("=" * 50)
    print(f"YOLO dataset created at: {args.output_dir}")
    print("\nDataset structure:")
    print("yolo_dataset/")
    print("├── rdd2022/          # RDD2022 dataset")
    print("│   ├── train/")
    print("│   └── val/")
    print("├── mwpd/             # MWPD dataset")
    print("│   ├── train/")
    print("│   └── val/")
    print("├── train/            # Combined dataset (if --create-combined)")
    print("├── val/")
    print("├── rdd_dataset.yaml  # RDD2022 only (4 classes)")
    print("├── mwpd_dataset.yaml # MWPD only (1 class)")
    print("└── rdd_mwpd.yaml     # Combined (4 classes, potholes merged)")
    print("\nReady for YOLO training!")


if __name__ == "__main__":
    main()