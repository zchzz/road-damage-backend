#!/usr/bin/env python3
"""
Model Evaluation Script
Based on oracl4/RoadDamageDetection/training/2_EvaluationTesting.ipynb
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, dataset_yaml: str, save_results: bool = True):
    """
    Evaluate trained model on validation dataset

    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset YAML file
        save_results: Whether to save evaluation results
    """

    print("Road Damage Detection Model Evaluation")
    print("=" * 50)

    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return None

    if not Path(dataset_yaml).exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        return None

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_yaml}")

    # Load model
    logger.info("Loading model...")
    model = YOLO(model_path)

    # Run evaluation
    logger.info("Running evaluation...")
    print("\nEvaluating model on validation set...")

    try:
        metrics = model.val(data=dataset_yaml, save=save_results)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        if hasattr(metrics, "box"):
            print("Overall Performance:")
            print(f"  mAP50: {metrics.box.map50:.3f}")
            print(f"  mAP50-95: {metrics.box.map:.3f}")
            print(f"  Precision: {metrics.box.mp:.3f}")
            print(f"  Recall: {metrics.box.mr:.3f}")

            # Per-class performance
            if hasattr(metrics.box, "maps") and len(metrics.box.maps) > 0:
                print("\nPer-Class Performance (mAP50):")
                class_names = [
                    "Longitudinal Crack",
                    "Transverse Crack",
                    "Alligator Crack",
                    "Potholes",
                ]

                for i, (class_name, map_score) in enumerate(
                    zip(class_names, metrics.box.maps)
                ):
                    if i < len(metrics.box.maps):
                        print(f"  {class_name}: {map_score:.3f}")

        # Speed information
        if hasattr(metrics, "speed"):
            print("\nInference Speed:")
            if "preprocess" in metrics.speed:
                print(f"  Preprocess: {metrics.speed['preprocess']:.1f}ms")
            if "inference" in metrics.speed:
                print(f"  Inference: {metrics.speed['inference']:.1f}ms")
            if "postprocess" in metrics.speed:
                print(f"  Postprocess: {metrics.speed['postprocess']:.1f}ms")

        if save_results:
            print("\nResults saved to: runs/detect/val")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def main():
    """Main evaluation workflow"""
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 model")

    parser.add_argument(
        "--model", required=True, help="Path to trained model weights (.pt file)"
    )
    parser.add_argument(
        "--dataset-yaml", required=True, help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save evaluation results"
    )

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model,
        dataset_yaml=args.dataset_yaml,
        save_results=not args.no_save,
    )

    if metrics is None:
        return 1

    print("\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
