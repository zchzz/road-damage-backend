#!/usr/bin/env python3
"""
YouTube Road Damage Analysis Script
Analyzes YouTube videos for road damage detection using YOLO
"""

import sys
import argparse
import re
from road_damage_detector import RoadDamageDetector
import logging


def extract_youtube_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([^&\n?#]+)",
        r"youtube\.com/watch\?.*v=([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze YouTube videos for road damage"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", help="Path to custom YOLO model (optional)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default="youtube_analysis",
        help="Output directory (default: youtube_analysis)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize detector
        detector = RoadDamageDetector(args.model)

        print(f"Analyzing YouTube video: {args.url}")
        print(f"Confidence threshold: {args.confidence}")
        print(f"Output directory: {args.output}")
        print("-" * 50)

        # Analyze video
        results = detector.analyze_youtube_video(args.url, args.output, args.confidence)

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total detections: {results['total_detections']}")
        print(f"Video duration: {results['video_duration']:.1f} seconds")
        print(f"Damage density: {results['damage_density']:.2f} damages/second")

        if "class_distribution" in results:
            print("\nDamage types detected:")
            for damage_type, count in results["class_distribution"].items():
                print(f"  {damage_type}: {count}")

        # Generate report
        video_id = extract_youtube_video_id(args.url)
        if video_id:
            report_filename = f"youtube_damage_report_{video_id}.html"
        else:
            report_filename = "youtube_damage_report_unknown.html"

        report_path = f"{args.output}/{report_filename}"
        detector.generate_report(results, report_path)
        print(f"\nDetailed report: {report_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
