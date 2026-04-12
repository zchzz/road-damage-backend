#!/usr/bin/env python3
"""
Enhanced Training System for Road Damage Detection
Based on RoadDamageDetection repository implementation
"""

import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path
import logging
from typing import Optional, List
import yaml
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDD2022DatasetDownloader:
    """Download and prepare RDD2022 dataset"""

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Dataset URL
        self.dataset_url = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip"
        self.dataset_zip = self.base_dir / "RDD2022.zip"
        self.dataset_dir = self.base_dir / "RDD2022"

        # Damage classes (from the reference repository)
        self.damage_classes = {
            "D00": 0,  # Longitudinal Crack
            "D10": 1,  # Transverse Crack
            "D20": 2,  # Alligator Crack
            "D40": 3,  # Potholes
        }

        self.class_names = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]

    def download_dataset(self, force_download: bool = False) -> bool:
        """Download RDD2022 dataset"""

        if self.dataset_zip.exists() and not force_download:
            logger.info(f"Dataset already exists: {self.dataset_zip}")
            return True

        logger.info(f"Downloading RDD2022 dataset from {self.dataset_url}")
        logger.info("This may take a while (dataset is ~3GB)")

        try:
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(self.dataset_zip, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Dataset downloaded successfully: {self.dataset_zip}")
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            if self.dataset_zip.exists():
                self.dataset_zip.unlink()
            return False

    def extract_dataset(self, force_extract: bool = False) -> bool:
        """Extract RDD2022 dataset"""

        if self.dataset_dir.exists() and not force_extract:
            logger.info(f"Dataset already extracted: {self.dataset_dir}")
            return True

        if not self.dataset_zip.exists():
            logger.error("Dataset zip file not found. Please download first.")
            return False

        logger.info(f"Extracting dataset to {self.dataset_dir}")

        try:
            with zipfile.ZipFile(self.dataset_zip, "r") as zip_ref:
                # Get total number of files for progress bar
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Extracting") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, self.base_dir)
                        pbar.update(1)

            logger.info("Dataset extracted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            return False

    def get_available_countries(self) -> List[str]:
        """Get list of available countries in the dataset"""
        countries_dir = self.dataset_dir / "RDD2022_all_countries"

        if not countries_dir.exists():
            logger.warning("Dataset not found or not extracted")
            return []

        countries = []
        for item in countries_dir.iterdir():
            if item.is_dir():
                countries.append(item.name)

        return sorted(countries)

    def prepare_for_training(
        self, countries: List[str] = None, output_dir: str = "training_data"
    ) -> str:
        """Prepare dataset for YOLO training"""

        if countries is None:
            # Use Japan and India as default (like in the reference repo)
            countries = ["Japan", "India"]

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Preparing training data for countries: {countries}")

        # Create YOLO dataset structure
        for country in countries:
            country_dir = output_path / country

            # Create directories
            for split in ["train", "val"]:
                (country_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (country_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Create dataset.yaml file
        yaml_content = {
            "path": str(output_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        yaml_file = output_path / "dataset.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        logger.info(f"Dataset configuration saved to: {yaml_file}")
        return str(yaml_file)


def download_pretrained_model(output_dir: str = "models") -> str:
    """Download pre-trained RDD model from the reference repository"""

    models_dir = Path(output_dir)
    models_dir.mkdir(exist_ok=True)

    model_url = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
    model_path = models_dir / "YOLOv8_Small_RDD.pt"

    if model_path.exists():
        logger.info(f"Pre-trained model already exists: {model_path}")
        return str(model_path)

    logger.info("Downloading pre-trained RDD model...")

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(model_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading model"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Pre-trained model downloaded: {model_path}")
        return str(model_path)

    except Exception as e:
        logger.error(f"Failed to download pre-trained model: {e}")
        return ""


def main():
    """Main function for dataset preparation"""
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare RDD2022 dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--extract", action="store_true", help="Extract dataset")
    parser.add_argument("--prepare", action="store_true", help="Prepare for training")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["Japan", "India"],
        help="Countries to include in training (default: Japan India)",
    )
    parser.add_argument(
        "--output-dir",
        default="training_data",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--models-dir", default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download, extract, and prepare dataset"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = RDD2022DatasetDownloader()

    if args.all or args.download:
        if not downloader.download_dataset():
            logger.error("Failed to download dataset")
            sys.exit(1)

    if args.all or args.extract:
        if not downloader.extract_dataset():
            logger.error("Failed to extract dataset")
            sys.exit(1)

    if args.all or args.prepare:
        # Show available countries
        available_countries = downloader.get_available_countries()
        logger.info(f"Available countries: {available_countries}")

        # Prepare dataset
        yaml_file = downloader.prepare_for_training(args.countries, args.output_dir)
        logger.info(f"Dataset prepared for training: {yaml_file}")

    # Download pre-trained model
    model_path = download_pretrained_model(args.models_dir)
    if model_path:
        logger.info(f"Pre-trained model available at: {model_path}")

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print("Next steps:")
    print("1. Convert XML annotations to YOLO format")
    print("2. Run training script")
    print("3. Evaluate results")


if __name__ == "__main__":
    main()
