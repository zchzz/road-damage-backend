#!/usr/bin/env python3
"""
Debug script to test what's happening with Gradio components
"""

import os
import json
import tempfile
from road_damage_detector import RoadDamageDetector


def test_detection_results():
    """Test if detection results contain proper data"""
    detector = RoadDamageDetector()

    # Test with a YouTube video
    test_url = "https://www.youtube.com/watch?v=C0007"  # The video from logs

    print("Testing YouTube analysis...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            results = detector.analyze_youtube_video(
                test_url,
                output_dir=temp_dir,
                confidence=0.3,
                stream_only=True,  # This is what the Gradio app uses by default
            )

            print(f"Results keys: {list(results.keys())}")
            print(f"Total detections: {results.get('total_detections', 0)}")
            print(f"Number of detection objects: {len(results.get('detections', []))}")

            # Print first few detections to see structure
            detections = results.get("detections", [])
            for i, detection in enumerate(detections[:3]):
                print(f"Detection {i}: {detection}")

            # Check if we have class distribution
            class_dist = results.get("class_distribution", {})
            print(f"Class distribution: {class_dist}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            print(traceback.format_exc())


if __name__ == "__main__":
    test_detection_results()
