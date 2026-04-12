#!/usr/bin/env python3
"""
Road Damage Detection Training Script
Based on oracl4/RoadDamageDetection implementation
Adapted for RDD2022 dataset with proper class mapping
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadDamageTrainer:
    def __init__(self, base_model: str = "yolov8s.pt"):
        """
        Initialize the Road Damage Trainer

        Args:
            base_model: Base YOLO model to start training from
                       Default: yolov8s.pt (same as reference implementation)
        """
        self.base_model = base_model
        self.model = None  # Initialize when needed

        # Road damage classes (from RDD2022 dataset - same as reference repo)
        self.class_names = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]

        logger.info(f"Initialized trainer with {base_model}")
        logger.info(f"Classes: {self.class_names}")

    def train_model(
        self,
        dataset_yaml: str,
        project: str = "runs/RDD_Training",
        name: str = "YOLOv8_RDD_Model",
        epochs: int = 100,
        batch: int = 32,
        imgsz: int = 640,
        workers: int = 1,
        device: str = "cpu",
        patience: int = 50,
        save_period: int = 10,
        cos_lr: bool = True,
        mosaic: float = 1.0,
        **kwargs,
    ):
        """
        Train YOLO model on road damage dataset
        Based on the reference implementation parameters

        Args:
            dataset_yaml: Path to dataset YAML file
            project: Project name for saving results
            name: Run name for this training session
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size for training
            workers: Number of data loading workers
            device: Device to use ('cpu', 'cuda', 'mps')
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            cos_lr: Use cosine learning rate scheduler
            mosaic: Mosaic augmentation probability
            **kwargs: Additional training parameters
        """

        # Initialize model
        logger.info(f"Loading base model: {self.base_model}")
        self.model = YOLO(self.base_model)

        # Calculate warmup epochs (5% of total epochs, same as reference)
        warmup_epochs = int(epochs * 0.05)

        logger.info("Starting training with following parameters:")
        logger.info(f"  Dataset: {dataset_yaml}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch}")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Device: {device}")

        # Training parameters (adapted from reference implementation)
        train_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "batch": batch,
            "imgsz": imgsz,
            "save_period": save_period,
            "workers": workers,
            "project": project,
            "name": name,
            "seed": 1337,  # For reproducibility
            "cos_lr": cos_lr,
            "patience": patience,
            "save": True,
            "cache": True,
            "device": device,
            "mosaic": mosaic,
            **kwargs,
        }

        # Start training
        try:
            results = self.model.train(**train_params)
            logger.info("Training completed successfully!")

            # Log final results
            if hasattr(results, "results_dict"):
                logger.info(f"Final metrics: {results.results_dict}")

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def validate_model(self, dataset_yaml: str, **kwargs):
        """
        Validate trained model

        Args:
            dataset_yaml: Path to dataset YAML file
            **kwargs: Additional validation parameters
        """
        if self.model is None:
            logger.error(
                "No model loaded. Train a model first or load existing weights."
            )
            return None

        logger.info("Evaluating model...")

        try:
            metrics = self.model.val(data=dataset_yaml, **kwargs)
            logger.info("Validation completed!")
            return metrics
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def load_model(self, model_path: str):
        """Load pre-trained model"""
        logger.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        return self.model

    def predict(self, source, **kwargs):
        """Make predictions with trained model"""
        if self.model is None:
            logger.error(
                "No model loaded. Train a model first or load existing weights."
            )
            return None

        return self.model.predict(source=source, **kwargs)


def main():
    """Main training workflow"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RDD2022 dataset")

    # Dataset arguments
    parser.add_argument(
        "--dataset-yaml",
        required=True,
        help="Path to dataset YAML file (from convert_dataset.py)",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8s.pt",
        help="Base YOLO model (default: yolov8s.pt)",
    )

    # Training arguments (with defaults from reference implementation)
    parser.add_argument(
        "--project",
        default="runs/RDD_Training",
        help="Project directory for saving results",
    )
    parser.add_argument(
        "--name", default="YOLOv8_RDD_Model", help="Run name for this training session"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for training"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--save-period", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # Training hyperparameters
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduler",
    )
    parser.add_argument(
        "--mosaic", type=float, default=1.0, help="Mosaic augmentation probability"
    )

    # Action arguments
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation (requires --model-path)",
    )
    parser.add_argument("--model-path", help="Path to pre-trained model for validation")

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.dataset_yaml).exists():
        logger.error(f"Dataset YAML not found: {args.dataset_yaml}")
        logger.info("Please run download_dataset.py and convert_dataset.py first")
        return 1

    if args.validate_only and not args.model_path:
        logger.error("--model-path required when using --validate-only")
        return 1

    # Initialize trainer
    trainer = RoadDamageTrainer(base_model=args.base_model)

    if args.validate_only:
        # Load model and validate
        trainer.load_model(args.model_path)
        metrics = trainer.validate_model(args.dataset_yaml)
        if metrics:
            logger.info("Validation completed successfully!")
        return 0

    # Training workflow
    logger.info("Starting training workflow...")

    try:
        # Train model
        trainer.train_model(
            dataset_yaml=args.dataset_yaml,
            project=args.project,
            name=args.name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            save_period=args.save_period,
            cos_lr=args.cos_lr,
            mosaic=args.mosaic,
        )

        # Validate after training
        logger.info("Running post-training validation...")
        metrics = trainer.validate_model(args.dataset_yaml)

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        print(f"Results saved to: {args.project}/{args.name}")
        print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
        print(f"Last weights: {args.project}/{args.name}/weights/last.pt")

        if metrics:
            print("\nValidation metrics:")
            if hasattr(metrics, "box"):
                print(f"mAP50: {metrics.box.map50:.3f}")
                print(f"mAP50-95: {metrics.box.map:.3f}")

        print("\nTo use the trained model:")
        print(
            f"python road_damage_detector.py --model {args.project}/{args.name}/weights/best.pt --source <video_or_image>"
        )

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
