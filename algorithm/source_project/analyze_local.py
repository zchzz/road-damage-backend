#!/usr/bin/env python3
"""
Local Video Analysis Script
Analyzes local video files for road damage detection using YOLO
"""

import sys
import argparse
import os
from road_damage_detector import RoadDamageDetector
import logging


def main():
    parser = argparse.ArgumentParser(description="Analyze local videos for road damage")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model", help="Path to custom YOLO model (optional)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="Process every nth frame (default: 5)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        # Initialize detector
        detector = RoadDamageDetector(args.model)

        # Set output path if not provided
        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.video_path))[0]
            args.output = f"analyzed_{base_name}.mp4"

        print(f"Analyzing video: {args.video_path}")
        print(f"Confidence threshold: {args.confidence}")
        print(f"Skip frames: {args.skip_frames}")
        print(f"Output video: {args.output}")
        print("-" * 50)

        # Analyze video
        results = detector.analyze_video(
            args.video_path, args.output, args.confidence, args.skip_frames
        )

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total detections: {results['total_detections']}")
        print(f"Video duration: {results['video_duration']:.1f} seconds")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Damage density: {results['damage_density']:.2f} damages/second")

        if "class_distribution" in results:
            print("\nDamage types detected:")
            for damage_type, count in results["class_distribution"].items():
                print(f"  {damage_type}: {count}")

        # Generate report
        report_path = f"analysis_report_{os.path.splitext(os.path.basename(args.video_path))[0]}.html"
        detector.generate_report(results, report_path)
        print(f"\nDetailed report: {report_path}")
        print(f"Annotated video: {args.output}")

        # Save results as JSON
        import json

        json_path = f"analysis_results_{os.path.splitext(os.path.basename(args.video_path))[0]}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Raw results: {json_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
