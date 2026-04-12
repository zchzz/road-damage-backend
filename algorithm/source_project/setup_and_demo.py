#!/usr/bin/env python3
"""
Road Damage Detection Setup and Demo Script
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path


def setup_environment():
    """Install required dependencies using uv in a virtual environment"""
    print("Setting up Road Damage Detection environment with uv...")

    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ uv is not installed. Please install uv first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  or visit: https://github.com/astral-sh/uv")
        return False

    venv_path = ".venv"

    try:
        # Create virtual environment with uv
        if not os.path.exists(venv_path):
            print("Creating virtual environment with uv...")
            subprocess.check_call(["uv", "venv", venv_path])
            print(f"✓ Virtual environment created at {venv_path}/")
        else:
            print(f"✓ Virtual environment already exists at {venv_path}/")

        # Install requirements using uv
        print("Installing dependencies with uv...")
        subprocess.check_call(["uv", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")

        # Show activation instructions
        print("\n" + "=" * 50)
        print("VIRTUAL ENVIRONMENT SETUP COMPLETE")
        print("=" * 50)
        print("To activate the virtual environment in the future, run:")
        if os.name == "nt":  # Windows
            print(f"  {venv_path}\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print(f"  source {venv_path}/bin/activate")
        print("\nOr use uv to run commands directly:")
        print("  uv run python script.py")
        print("=" * 50)

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to setup environment: {e}")
        print("Falling back to regular pip installation...")

        # Fallback to regular pip
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            )
            print("✓ Dependencies installed with pip (fallback)")
            return True
        except subprocess.CalledProcessError as fallback_error:
            print(f"✗ Fallback installation also failed: {fallback_error}")
            return False


def analyze_local_video(video_path: str, confidence: float = 0.3, skip_frames: int = 5):
    """Analyze local video file"""
    print("\n" + "=" * 50)
    print("LOCAL VIDEO ANALYSIS")
    print("=" * 50)

    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return False

    print(f"Analyzing local video: {video_path}")
    print(f"Confidence threshold: {confidence}")
    print(f"Skip frames: {skip_frames}")

    try:
        from road_damage_detector import RoadDamageDetector

        # Initialize detector
        detector = RoadDamageDetector()

        # Set output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video = f"analyzed_{base_name}.mp4"

        # Analyze video
        results = detector.analyze_video(
            video_path,
            output_path=output_video,
            confidence=confidence,
            skip_frames=skip_frames,
        )

        print("✓ Analysis complete!")
        print(f"  Total detections: {results['total_detections']}")
        print(f"  Video duration: {results['video_duration']:.1f} seconds")
        print(f"  Damage density: {results['damage_density']:.2f} damages/second")

        # Generate report
        report_path = f"report_{base_name}.html"
        detector.generate_report(results, report_path)
        print(f"✓ Report generated: {report_path}")
        print(f"✓ Annotated video: {output_video}")

        if "class_distribution" in results and results["class_distribution"]:
            print("\nDamage types detected:")
            for damage_type, count in results["class_distribution"].items():
                print(f"  {damage_type}: {count}")

        if results["total_detections"] > 0:
            print("\nFirst 5 detections:")
            for i, detection in enumerate(results["detections"][:5]):
                print(
                    f"  {i+1}. {detection['class']} at {detection['timestamp']:.1f}s (conf: {detection['confidence']:.2f})"
                )

        return True

    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return False


def analyze_youtube_video(
    youtube_url: str, confidence: float = 0.3, output_dir: str = "youtube_analysis"
):
    """Analyze YouTube video"""
    print("\n" + "=" * 50)
    print("YOUTUBE VIDEO ANALYSIS")
    print("=" * 50)

    # Check if using default URL
    default_url = "https://youtube.com/@uptdpjjkertobaworo?si=LBFiWHV7pRKUoVQW"
    if youtube_url == default_url:
        print("Using default YouTube channel for road damage analysis")

    print(f"YouTube URL: {youtube_url}")
    print(f"Confidence threshold: {confidence}")
    print(f"Output directory: {output_dir}")

    try:
        from road_damage_detector import RoadDamageDetector

        # Initialize detector
        detector = RoadDamageDetector()

        # Analyze YouTube video
        results = detector.analyze_youtube_video(youtube_url, output_dir, confidence)

        print("✓ Analysis complete!")
        print(f"  Total detections: {results['total_detections']}")
        print(f"  Video duration: {results['video_duration']:.1f} seconds")
        print(f"  Damage density: {results['damage_density']:.2f} damages/second")

        if "class_distribution" in results and results["class_distribution"]:
            print("\nDamage types detected:")
            for damage_type, count in results["class_distribution"].items():
                print(f"  {damage_type}: {count}")

        if results["total_detections"] > 0:
            print("\nFirst 5 detections:")
            for i, detection in enumerate(results["detections"][:5]):
                print(
                    f"  {i+1}. {detection['class']} at {detection['timestamp']:.1f}s (conf: {detection['confidence']:.2f})"
                )

        print(f"\nResults saved to: {output_dir}/")
        return True

    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return False


def demo_training_setup():
    """Demo training data preparation"""
    print("\n" + "=" * 50)
    print("DEMO: Training Data Preparation")
    print("=" * 50)

    video_path = "/home/bagus/github/road-damage-detection/15-01-2024/C0007.MP4"

    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return

    print(f"Preparing training data from: {video_path}")

    try:
        from train_model import RoadDamageTrainer

        # Initialize trainer
        trainer = RoadDamageTrainer()

        # Create dataset (extract frames)
        dataset_yaml = trainer.create_training_dataset(video_path, "demo_dataset")

        print(f"✓ Training dataset prepared!")
        print(f"  Dataset config: {dataset_yaml}")
        print(f"  Frames extracted to: demo_dataset/")

        # Create sample annotations for demo
        trainer.create_sample_annotations("demo_dataset")
        print(f"✓ Sample annotations created (for demo purposes)")

        print("\nNext steps for real training:")
        print("1. Manually annotate the extracted frames")
        print("2. Use tools like LabelImg or Roboflow")
        print("3. Run training with proper annotations")

    except Exception as e:
        print(f"✗ Training setup failed: {e}")


def show_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES")
    print("=" * 50)

    print("\n1. Setup and demo (default):")
    print("   python setup_and_demo.py")
    print("   python setup_and_demo.py --setup-only")

    print("\n2. Analyze local video:")
    print("   python setup_and_demo.py --local /path/to/video.mp4")
    print("   python setup_and_demo.py --local video.mp4 --confidence 0.5")
    print("   python setup_and_demo.py --local video.mp4 --conf 0.3 --skip-frames 10")

    print("\n3. Analyze YouTube video:")
    print(
        "   python setup_and_demo.py --youtube                           # Use default channel"
    )
    print(
        "   python setup_and_demo.py --youtube 'https://youtube.com/watch?v=VIDEO_ID'"
    )
    print(
        "   python setup_and_demo.py --youtube 'https://youtu.be/VIDEO_ID' --conf 0.4"
    )
    print("   python setup_and_demo.py --youtube URL --output-dir my_analysis")

    print("\n4. Alternative dedicated scripts:")
    print("   python analyze_youtube.py 'https://youtube.com/watch?v=VIDEO_ID'")
    print("   python analyze_local.py /path/to/video.mp4")

    print("\n5. Train custom model:")
    print("   python train_model.py  # Prepare training data")
    print("   # Then manually annotate the data")
    print("   # Training will run automatically with sample data")

    print("\n6. Use in Python code:")
    print("""   from road_damage_detector import RoadDamageDetector
   
   detector = RoadDamageDetector()
   results = detector.analyze_video('video.mp4', confidence=0.3)
   detector.generate_report(results, 'report.html')""")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Road Damage Detection Setup and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_and_demo.py                                    # Setup only
  python setup_and_demo.py --local /path/to/video.mp4        # Analyze local video
  python setup_and_demo.py --youtube                         # Analyze default YouTube channel
  python setup_and_demo.py --youtube https://youtube.com/... # Analyze specific YouTube video
  python setup_and_demo.py --local video.mp4 --conf 0.5     # Custom confidence
        """,
    )

    # Analysis mode arguments
    analysis_group = parser.add_mutually_exclusive_group()
    analysis_group.add_argument(
        "--local", metavar="PATH", help="Analyze local video file"
    )
    analysis_group.add_argument(
        "--youtube",
        nargs="?",
        const="https://youtube.com/@uptdpjjkertobaworo?si=LBFiWHV7pRKUoVQW",
        metavar="URL",
        help="Analyze YouTube video (default: road damage channel if no URL provided)",
    )

    # Optional arguments
    parser.add_argument(
        "--confidence",
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        help="Process every nth frame (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Output directory for YouTube analysis (default: analysis_output)",
    )
    parser.add_argument(
        "--setup-only", action="store_true", help="Only run setup and demo, no analysis"
    )

    args = parser.parse_args()

    print("Road Damage Detection System")
    print("=" * 50)

    # Check if running from correct directory
    if not os.path.exists("requirements.txt"):
        print("✗ Please run this script from the project root directory")
        sys.exit(1)

    # Setup environment
    if not setup_environment():
        print("✗ Setup failed. Please check your Python environment.")
        sys.exit(1)

    # If no analysis arguments provided or setup-only flag, run setup demo
    if not args.local and not args.youtube or args.setup_only:
        print("\n" + "=" * 50)
        print("RUNNING SETUP DEMO")
        print("=" * 50)

        # Demo with default video if it exists
        default_video = "/home/bagus/github/road-damage-detection/15-01-2024/C0007.MP4"
        if os.path.exists(default_video):
            print("Running demo with default video...")
            analyze_local_video(default_video, confidence=0.1, skip_frames=10)

        # Run training setup demo
        demo_training_setup()

        # Show usage examples
        show_usage_examples()

        print("\n" + "=" * 50)
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("✓ Environment configured")
        if os.path.exists(default_video):
            print("✓ Demo analysis completed")
        print("✓ Training data prepared")
        print("\nThe system is ready to use!")

    elif args.local:
        # Analyze local video
        success = analyze_local_video(args.local, args.confidence, args.skip_frames)
        if not success:
            sys.exit(1)

    elif args.youtube:
        # Analyze YouTube video
        success = analyze_youtube_video(args.youtube, args.confidence, args.output_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
