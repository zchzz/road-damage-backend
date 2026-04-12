#!/usr/bin/env python3
"""
Demo script to test the Road Damage Detection system
"""

import os
from road_damage_detector import RoadDamageDetector

def demo_local_analysis():
    """Demo local video analysis"""
    print("🚀 Road Damage Detection Demo")
    print("=" * 50)
    
    # Check for local video file
    video_path = "15-01-2024/C0007.MP4"
    if not os.path.exists(video_path):
        print("❌ Demo video not found at:", video_path)
        print("Please ensure you have a video file to analyze.")
        return
    
    print(f"📹 Analyzing video: {video_path}")
    print("🔧 Initializing detector...")
    
    try:
        # Initialize detector
        detector = RoadDamageDetector()
        
        print("🔍 Running analysis...")
        print("⚙️  Using confidence threshold: 0.3")
        print("⚙️  Processing every 5th frame for speed")
        
        # Analyze video
        results = detector.analyze_video(
            video_path,
            output_path="demo_output.mp4",
            confidence=0.3,
            skip_frames=5
        )
        
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"🎯 Total Detections: {results['total_detections']}")
        print(f"⏱️  Video Duration: {results['video_duration']:.1f} seconds")
        print(f"📈 Damage Density: {results['damage_density']:.3f} damages/second")
        print(f"🖼️  Processed Frames: {results['processed_frames']:,}")
        
        if results.get('class_distribution'):
            print("\n🏷️  Damage Types Found:")
            for damage_type, count in results['class_distribution'].items():
                percentage = (count / results['total_detections']) * 100
                print(f"   • {damage_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n📁 Output Files:")
        print(f"   • Annotated video: demo_output.mp4")
        
        # Generate report
        detector.generate_report(results, "demo_report.html")
        print(f"   • HTML report: demo_report.html")
        
        print("\n✅ Demo completed successfully!")
        print("🌐 You can now run the Gradio app with: python gradio_app.py")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def demo_youtube_analysis():
    """Demo YouTube video analysis"""
    print("\n📺 YouTube Analysis Demo")
    print("-" * 30)
    
    # Example road inspection video URLs (replace with actual ones)
    sample_urls = [
        "https://www.youtube.com/watch?v=SAMPLE_URL_1",
        "https://www.youtube.com/watch?v=SAMPLE_URL_2"
    ]
    
    print("To test YouTube analysis, you can use URLs like:")
    for i, url in enumerate(sample_urls, 1):
        print(f"  {i}. {url}")
    
    print("\nNote: Use actual YouTube URLs with road footage for real testing")
    print("The Gradio app provides an easy interface for YouTube analysis")

if __name__ == "__main__":
    demo_local_analysis()
    demo_youtube_analysis()
