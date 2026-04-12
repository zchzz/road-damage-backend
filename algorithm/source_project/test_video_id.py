#!/usr/bin/env python3
"""
Test script for YouTube video ID extraction
"""

import re


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


# Test cases
test_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmRdnEQy",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
]

print("Testing YouTube Video ID Extraction:")
print("=" * 50)

for url in test_urls:
    video_id = extract_youtube_video_id(url)
    print(f"URL: {url}")
    print(f"Video ID: {video_id}")
    print("-" * 30)

print("Expected Video ID for all tests: dQw4w9WgXcQ")
