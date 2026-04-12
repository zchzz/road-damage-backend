#!/usr/bin/env python3
"""
Test script for Road Damage Detection System
Runs various tests to ensure the system is working correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import cv2

        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        from ultralytics import YOLO

        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False

    try:
        from pytube import YouTube

        print("✓ PyTube imported successfully")
    except ImportError as e:
        print(f"✗ PyTube import failed: {e}")
        return False

    try:
        import torch

        print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available with {torch.cuda.device_count()} device(s)")
        else:
            print("! CUDA not available, will use CPU")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    return True


def test_video_loading():
    """Test if the sample video can be loaded"""
    print("\nTesting video loading...")

    video_path = "/home/bagus/github/road-damage-detection/15-01-2024/C0007.MP4"

    if not os.path.exists(video_path):
        print(f"✗ Sample video not found: {video_path}")
        return False

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        cap.release()

        print(f"✓ Video loaded successfully")
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {int(frame_count)}")
        print(f"  Duration: {frame_count/fps:.1f} seconds")

        return True

    except Exception as e:
        print(f"✗ Video loading failed: {e}")
        return False


def test_yolo_model():
    """Test YOLO model loading and inference"""
    print("\nTesting YOLO model...")

    try:
        from ultralytics import YOLO
        import numpy as np

        # Load model
        model = YOLO("yolov8n.pt")
        print("✓ YOLOv8n model loaded successfully")

        # Test inference with dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)

        print("✓ Model inference test successful")
        return True

    except Exception as e:
        print(f"✗ YOLO model test failed: {e}")
        return False


def test_detector_class():
    """Test the RoadDamageDetector class"""
    print("\nTesting RoadDamageDetector class...")

    try:
        from road_damage_detector import RoadDamageDetector

        # Initialize detector
        detector = RoadDamageDetector()
        print("✓ RoadDamageDetector initialized successfully")

        # Test frame detection with dummy data
        import numpy as np

        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        annotated_frame, detections = detector.detect_frame(dummy_frame, confidence=0.5)
        print(f"✓ Frame detection test successful (found {len(detections)} detections)")

        return True

    except Exception as e:
        print(f"✗ RoadDamageDetector test failed: {e}")
        return False


def test_training_class():
    """Test the RoadDamageTrainer class"""
    print("\nTesting RoadDamageTrainer class...")

    try:
        from train_model import RoadDamageTrainer

        # Initialize trainer
        trainer = RoadDamageTrainer()
        print("✓ RoadDamageTrainer initialized successfully")

        return True

    except Exception as e:
        print(f"✗ RoadDamageTrainer test failed: {e}")
        return False


def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")

    required_files = [
        "road_damage_detector.py",
        "train_model.py",
        "analyze_youtube.py",
        "analyze_local.py",
        "setup_and_demo.py",
        "requirements.txt",
        "config.py",
        "README.md",
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_exist = False

    return all_exist


def run_mini_analysis():
    """Run a mini analysis on the sample video"""
    print("\nRunning mini analysis...")

    video_path = "/home/bagus/github/road-damage-detection/15-01-2024/C0007.MP4"

    if not os.path.exists(video_path):
        print(f"✗ Sample video not found: {video_path}")
        return False

    try:
        from road_damage_detector import RoadDamageDetector

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_video = os.path.join(temp_dir, "test_output.mp4")

            # Initialize detector
            detector = RoadDamageDetector()

            # Analyze just first 60 frames for quick test
            import cv2

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("✗ Cannot open video for mini analysis")
                return False

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            total_detections = 0
            frame_count = 0

            while frame_count < 60:  # Process only 60 frames
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect damage
                annotated_frame, detections = detector.detect_frame(
                    frame, confidence=0.1
                )
                total_detections += len(detections)

                # Write frame
                out.write(annotated_frame)
                frame_count += 1

            cap.release()
            out.release()

            print(f"✓ Mini analysis completed")
            print(f"  Processed frames: {frame_count}")
            print(f"  Total detections: {total_detections}")
            print(f"  Output video size: {os.path.getsize(output_video)} bytes")

            return True

    except Exception as e:
        print(f"✗ Mini analysis failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Road Damage Detection System - Test Suite")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Video Loading", test_video_loading),
        ("YOLO Model", test_yolo_model),
        ("Detector Class", test_detector_class),
        ("Training Class", test_training_class),
        ("Mini Analysis", run_mini_analysis),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("Run 'pip install -r requirements.txt' if import errors occurred.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
