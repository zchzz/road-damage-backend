#!/usr/bin/env python3
"""
Unified Road Damage Analysis Script
Analyzes both local video files and YouTube videos for road damage detection using YOLO
Automatically detects source type based on input format
"""

import sys
import argparse
import os
import re
from road_damage_detector import RoadDamageDetector
import logging


def is_youtube_url(source: str) -> bool:
    """Check if the source is a YouTube URL"""
    youtube_patterns = [
        r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/",
        r"(https?://)?(m\.)?youtube\.com/",
        r"(https?://)?youtu\.be/",
        r"^[a-zA-Z0-9_-]{11}$",  # Direct video ID
    ]

    for pattern in youtube_patterns:
        if re.search(pattern, source, re.IGNORECASE):
            return True
    return False


def normalize_youtube_url(source: str) -> str:
    """Normalize YouTube URL or video ID to full URL"""
    # If it's just a video ID (11 characters)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", source):
        return f"https://youtube.com/watch?v={source}"

    # If it's already a full URL, return as-is
    if source.startswith(("http://", "https://")):
        return source

    # If it's a youtube.com path without protocol
    if "youtube.com" in source or "youtu.be" in source:
        return (
            f"https://{source}"
            if not source.startswith("www.")
            else f"https://{source}"
        )

    return source


def main():
    parser = argparse.ArgumentParser(
        description="Analyze videos (local or YouTube) for road damage detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local video file
  python analyze.py --source /path/to/video.mp4

  # YouTube URL (download and analyze)
  python analyze.py --source "https://youtube.com/watch?v=VIDEO_ID"

  # YouTube URL (stream without downloading)
  python analyze.py --source "https://youtube.com/watch?v=VIDEO_ID" --stream

  # YouTube video ID only (streaming mode)
  python analyze.py --source VIDEO_ID --stream

  # With custom parameters
  python analyze.py --source VIDEO_ID --confidence 0.5 --model custom_model.pt --stream
        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source: local video file path or YouTube URL/video ID",
    )
    parser.add_argument("--model", help="Path to custom YOLO model (optional)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output", help="Output path/directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="Process every nth frame for local videos (default: 5)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream YouTube videos without downloading (analysis only, no output video)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize detector
        detector = RoadDamageDetector(args.model)

        # Determine source type
        if is_youtube_url(args.source):
            youtube_url = normalize_youtube_url(args.source)

            print(f"🎥 Analyzing YouTube video: {youtube_url}")
            print(f"📊 Confidence threshold: {args.confidence}")

            if args.stream:
                print("🌊 Mode: Streaming (no download, analysis only)")
            else:
                print("💾 Mode: Download and analyze")

            # Set output directory
            output_dir = args.output if args.output else "youtube_analysis"
            print(f"📁 Output directory: {output_dir}")
            print("-" * 60)

            # Analyze video with streaming option
            results = detector.analyze_youtube_video(
                youtube_url, output_dir, args.confidence, stream_only=args.stream
            )

            # Generate report
            report_path = f"{output_dir}/youtube_damage_report.html"
            detector.generate_report(results, report_path)

            # Print summary
            print("\n" + "=" * 60)
            print("🎯 YOUTUBE ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"📊 Total detections: {results['total_detections']}")
            print(f"⏱️  Video duration: {results['video_duration']:.1f} seconds")
            print(f"📈 Damage density: {results['damage_density']:.2f} damages/second")

            if results.get("streaming"):
                print("🌊 Analysis method: Streaming (no local file created)")
            else:
                print("💾 Analysis method: Downloaded and analyzed")

            if "class_distribution" in results and results["class_distribution"]:
                print("\n🔍 Damage types detected:")
                for damage_type, count in results["class_distribution"].items():
                    print(f"   • {damage_type}: {count}")

            print(f"\n📋 Detailed report: {report_path}")

            if not args.stream:
                print(f"🎬 Annotated video: {output_dir}/analyzed_*.mp4")
            else:
                print("🌊 No output video (streaming mode)")

        else:
            # Local video analysis
            video_path = args.source

            # Check if video exists
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found: {video_path}")
                sys.exit(1)

            print(f"🎥 Analyzing local video: {video_path}")
            print(f"📊 Confidence threshold: {args.confidence}")
            print(f"⏭️  Skip frames: {args.skip_frames}")

            # Set output path
            if args.output:
                output_video = args.output
            else:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_video = f"analyzed_{base_name}.mp4"

            print(f"📁 Output video: {output_video}")
            print("-" * 60)

            # Analyze video
            results = detector.analyze_video(
                video_path,
                output_path=output_video,
                confidence=args.confidence,
                skip_frames=args.skip_frames,
            )

            # Generate report
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            report_path = f"local_damage_report_{base_name}.html"
            detector.generate_report(results, report_path)

            # Print summary
            print("\n" + "=" * 60)
            print("🎯 LOCAL VIDEO ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"📊 Total detections: {results['total_detections']}")
            print(f"⏱️  Video duration: {results['video_duration']:.1f} seconds")
            print(f"📈 Damage density: {results['damage_density']:.2f} damages/second")
            print(f"🖼️  Frames processed: {results.get('frames_processed', 'N/A')}")

            if "class_distribution" in results and results["class_distribution"]:
                print("\n🔍 Damage types detected:")
                for damage_type, count in results["class_distribution"].items():
                    print(f"   • {damage_type}: {count}")

            if results["total_detections"] > 0:
                print("\n📋 First 5 detections:")
                for i, detection in enumerate(results["detections"][:5]):
                    print(
                        f"   {i+1}. {detection['class']} at {detection['timestamp']:.1f}s "
                        f"(confidence: {detection['confidence']:.2f})"
                    )

            print(f"\n📋 Detailed report: {report_path}")
            print(f"🎬 Annotated video: {output_video}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
