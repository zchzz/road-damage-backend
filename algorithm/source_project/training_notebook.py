#!/usr/bin/env python3
"""
Training Notebook - Interactive Training Script
Based on oracl4/RoadDamageDetection/training/1_TrainingYOLOv8.ipynb
"""

from ultralytics import YOLO
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Interactive training similar to the reference notebook"""

    print("Road Damage Detection Training")
    print("=" * 50)

    # Check if dataset exists
    dataset_yaml = "yolo_dataset/rdd_dataset.yaml"

    if not Path(dataset_yaml).exists():
        print("ERROR: Dataset not found!")
        print("Please run the following commands first:")
        print("1. python download_dataset.py")
        print(
            "2. python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset"
        )
        return

    print(f"Using dataset: {dataset_yaml}")

    # Project settings (same as reference implementation)
    project = "runs/RDD_Training"
    name = "YOLOv8_RDD_Model"
    data = dataset_yaml

    # Training parameters (adapted from reference)
    epochs = 100
    warmup_epochs = int(epochs * 0.05)  # 5% of total epochs

    print("\nTraining Parameters:")
    print("  Model: yolov8s.pt")
    print(f"  Epochs: {epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print("  Batch size: 16")
    print("  Image size: 1280")
    print(f"  Project: {project}")
    print(f"  Name: {name}")

    # Initialize model
    print("\nLoading YOLOv8s model...")
    model = YOLO(r"C:\Users\Administrator\Desktop\road-damage-detection-master\models\best.pt")

    # Start training
    print("\nStarting training...")
    print("This may take a while depending on your hardware.")

    try:
        model.train(
            data=data,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch=16,
            imgsz=1280,
            save_period=10,
            workers=1,
            project=project,
            name=name,
            seed=1337,
            cos_lr=True,
            patience=0,
            mosaic=0.5,
        )

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Results saved to: {project}/{name}")
        print(f"Best weights: {project}/{name}/weights/best.pt")

        # Run validation
        print("\nRunning validation...")
        metrics = model.val(data=data)

        if hasattr(metrics, "box"):
            print("Validation Results:")
            print(f"  mAP50: {metrics.box.map50:.3f}")
            print(f"  mAP50-95: {metrics.box.map:.3f}")

    except Exception as e:
        print(f"Training failed: {e}")
        logger.error(f"Training error: {e}")


if __name__ == "__main__":
    main()
