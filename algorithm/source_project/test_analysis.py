#!/usr/bin/env python3
"""
Test road damage detection on a YouTube video
"""

from road_damage_detector import RoadDamageDetector
import sys


def test_youtube_analysis():
    """Test analyzing a YouTube video for road damage"""

    # Initialize detector
    detector = RoadDamageDetector()

    # Test video - this appears to be a road video from Indonesia
    test_url = "https://www.youtube.com/watch?v=YM773xuW3RU"

    print(f"Analyzing YouTube video: {test_url}")
    print("This may take a few minutes...")

    try:
        # Analyze with streaming (no download)
        results = detector.analyze_youtube_video(
            youtube_url=test_url,
            output_dir="test_output",
            confidence=0.5,
            stream_only=True,  # Use streaming mode
        )

        print("\n📊 Analysis Results:")
        print(f"Total frames processed: {results.get('total_frames', 'Unknown')}")
        print(f"Frames with damage: {results.get('frames_with_damage', 'Unknown')}")
        print(f"Total detections: {results.get('total_detections', 'Unknown')}")

        if "damage_types" in results:
            print("Damage types detected:")
            for damage_type, count in results["damage_types"].items():
                print(f"  - {damage_type}: {count}")

        print("✅ YouTube analysis test PASSED")
        return True

    except Exception as e:
        print(f"❌ YouTube analysis test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🛣️  Testing Road Damage Detection on YouTube Video")
    print("=" * 55)

    success = test_youtube_analysis()

    if success:
        print("\n🎉 YouTube road damage analysis is working!")
    else:
        print("\n⚠️  YouTube analysis failed. Check the error above.")
        sys.exit(1)
