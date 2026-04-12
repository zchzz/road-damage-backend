#!/usr/bin/env python3
"""
Test script to demonstrate YouTube streaming analysis
"""

from road_damage_detector import RoadDamageDetector
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_streaming():
    """Test YouTube streaming analysis"""

    # Example YouTube video URLs for testing (replace with actual road videos)
    test_urls = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ",  # Short test video
        # Add your own YouTube URLs here
    ]

    print("🎯 Testing YouTube Streaming Analysis")
    print("=" * 50)

    detector = RoadDamageDetector()

    for i, url in enumerate(test_urls, 1):
        print(f"\n📹 Test {i}: Streaming analysis")
        print(f"URL: {url}")
        print("-" * 30)

        try:
            # Stream analysis (no download)
            results = detector.analyze_youtube_video_stream(
                url,
                confidence=0.3,
                max_frames=300,  # Limit to first 300 frames for testing
            )

            print("✅ Streaming analysis complete!")
            print(f"   - Video: {results.get('video_title', 'Unknown')}")
            print(f"   - Duration: {results['video_duration']:.1f}s")
            print(f"   - Frames processed: {results['processed_frames']}")
            print(f"   - Damages found: {results['total_detections']}")

            if results.get("class_distribution"):
                print("   - Damage types:")
                for damage_type, count in results["class_distribution"].items():
                    print(f"     • {damage_type}: {count}")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print("\n" + "=" * 50)
    print("🎯 Test complete!")


if __name__ == "__main__":
    test_streaming()
