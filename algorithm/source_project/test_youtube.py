#!/usr/bin/env python3
"""
Test script to verify pytubefix YouTube functionality
"""

import sys
import os
import tempfile
from pytubefix import YouTube


def test_youtube_access():
    """Test basic YouTube access and streaming"""

    # Test with a public YouTube video
    test_url = "https://www.youtube.com/watch?v=YM773xuW3RU"  # Sample video

    try:
        print(f"Testing YouTube access with: {test_url}")

        # Create YouTube object
        yt = YouTube(test_url)

        # Get basic info
        print(f"Title: {yt.title}")
        print(f"Length: {yt.length} seconds")
        print(f"Views: {yt.views}")

        # Get available streams
        streams = yt.streams.filter(progressive=True, file_extension="mp4")
        print(f"Available streams: {len(streams)}")

        if streams:
            # Get the lowest quality stream for testing
            stream = streams.order_by("resolution").first()
            print(f"Selected stream: {stream.resolution} - {stream.filesize} bytes")

            # Test streaming URL access (don't download, just get URL)
            stream_url = stream.url
            print(f"Stream URL obtained successfully")
            print("✅ YouTube access test PASSED")
            return True
        else:
            print("❌ No suitable streams found")
            return False

    except Exception as e:
        print(f"❌ YouTube access test FAILED: {e}")
        return False


def test_road_damage_detector_import():
    """Test that our main module can be imported"""
    try:
        from road_damage_detector import RoadDamageDetector

        print("✅ RoadDamageDetector import test PASSED")
        return True
    except Exception as e:
        print(f"❌ RoadDamageDetector import test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing YouTube and Road Damage Detection Setup")
    print("=" * 50)

    success = True

    # Test YouTube access
    success &= test_youtube_access()
    print()

    # Test main module import
    success &= test_road_damage_detector_import()
    print()

    if success:
        print("🎉 All tests PASSED! Your setup is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some tests FAILED. Please check the errors above.")
        sys.exit(1)
