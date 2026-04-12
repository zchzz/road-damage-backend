#!/usr/bin/env python3
"""
Gradio Web Application for Road Damage Detection
Interactive interface for analyzing videos (local upload or YouTube URL)
and generating comprehensive reports with detection details and images.
"""

import gradio as gr
import pandas as pd
import tempfile
import os
import atexit
import shutil
import json
import logging
import traceback
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json
from io import BytesIO

from road_damage_detector import RoadDamageDetector


class RoadDamageGradioApp:
    def __init__(self):
        """Initialize the Gradio app"""
        self.detector = RoadDamageDetector()
        self.logger = logging.getLogger(__name__)

        # Create temporary directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix="gradio_road_damage_")
        self.logger.info(f"Temporary directory: {self.temp_dir}")

        # Create main output directory for organized results
        self.output_base_dir = os.path.join(os.getcwd(), "analysis_results")
        os.makedirs(self.output_base_dir, exist_ok=True)

        # Register cleanup on exit
        atexit.register(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        try:
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Could not clean up temporary directory: {e}")

    def create_organized_output_dir(self, video_id, video_type="video"):
        """Create organized output directory for a video ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            self.output_base_dir, f"{video_type}_{video_id}_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        subdirs = ["frames", "reports", "videos", "data"]
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        return output_dir

    def _deduplicate_detections(
        self, detections, frame_distance=30, spatial_threshold=0.7
    ):
        """
        Remove duplicate detections from nearby frames or similar locations

        Args:
            detections: List of detection dictionaries
            frame_distance: Minimum frame distance to consider detections separate
            spatial_threshold: IoU threshold for spatial overlap (0-1, higher = more strict)

        Returns:
            Deduplicated list of detections
        """
        if not detections:
            return []

        # Sort detections by confidence (highest first)
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )

        deduplicated = []

        for detection in sorted_detections:
            is_duplicate = False

            for existing in deduplicated:
                # Check frame distance
                frame_diff = abs(detection["frame"] - existing["frame"])

                # If frames are close, check spatial overlap
                if frame_diff <= frame_distance:
                    iou = self._calculate_iou(detection["bbox"], existing["bbox"])

                    # If high spatial overlap, consider it a duplicate
                    if iou > spatial_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(detection)

        return deduplicated

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # bbox format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def save_detection_frames_to_disk(
        self, video_path, detections, output_dir, max_images=20
    ):
        """Extract and save detection frames to disk with deduplication"""
        if not detections or not video_path or not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frames_dir = os.path.join(output_dir, "frames")
        saved_frame_paths = []
        images_for_gradio = []

        # Deduplicate detections to avoid multiple images from same location
        deduplicated_detections = self._deduplicate_detections(detections)

        # Sort deduplicated detections by confidence and take top ones
        sorted_detections = sorted(
            deduplicated_detections, key=lambda x: x["confidence"], reverse=True
        )[:max_images]

        for i, detection in enumerate(sorted_detections):
            frame_number = detection["frame"]

            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Create annotated frame
                annotated_frame = frame.copy()

                # Draw bounding box on frame
                bbox = detection["bbox"]
                cv2.rectangle(
                    annotated_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    3,
                )

                # Add label
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # Add timestamp
                timestamp_label = (
                    f"Time: {self.format_timestamp(detection['timestamp'])}"
                )
                cv2.putText(
                    annotated_frame,
                    timestamp_label,
                    (bbox[0], bbox[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Save annotated frame to disk
                frame_filename = (
                    f"detection_{i+1:03d}_frame_{frame_number}_{detection['class']}.jpg"
                )
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                saved_frame_paths.append(frame_path)

                # Convert BGR to RGB for Gradio display
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                images_for_gradio.append(Image.fromarray(frame_rgb))

        cap.release()
        return images_for_gradio, saved_frame_paths

    def format_timestamp(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def extract_detection_frames(self, video_path, detections, max_images=10):
        """Extract frames with detections for display"""
        if not detections or not video_path or not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        images = []

        # Sort detections by confidence and take top ones
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )[:max_images]

        for detection in sorted_detections:
            frame_number = detection["frame"]

            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Draw bounding box on frame
                bbox = detection["bbox"]
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3
                )

                # Add label
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # Add timestamp
                timestamp_label = (
                    f"Time: {self.format_timestamp(detection['timestamp'])}"
                )
                cv2.putText(
                    frame,
                    timestamp_label,
                    (bbox[0], bbox[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb))

        cap.release()
        return images

    def extract_and_save_frames_from_youtube(
        self, youtube_url, detections, output_dir, max_images=20
    ):
        """Extract and save frames with detections directly from YouTube video"""
        if not detections or not youtube_url:
            return [], []

        try:
            import tempfile
            from pytubefix import YouTube

            with tempfile.TemporaryDirectory() as temp_dir:
                # Download video to temporary location for frame extraction
                yt = YouTube(youtube_url)
                stream = yt.streams.filter(
                    progressive=True, file_extension="mp4"
                ).first()
                if not stream:
                    stream = yt.streams.filter(file_extension="mp4").first()

                if not stream:
                    self.logger.warning("No suitable stream found for YouTube video")
                    return [], []

                temp_video_path = stream.download(temp_dir)

                # Use existing method to extract and save frames
                return self.save_detection_frames_to_disk(
                    temp_video_path, detections, output_dir, max_images
                )

        except Exception as e:
            self.logger.error(f"Error extracting frames from YouTube: {e}")
            return [], []

    def create_detection_chart(self, results):
        """Create a chart showing damage types distribution"""
        if not results.get("class_distribution"):
            return None

        # Create pie chart
        plt.figure(figsize=(12, 5))
        class_dist = results["class_distribution"]

        # Pie chart
        plt.subplot(1, 2, 1)
        plt.pie(
            class_dist.values(),
            labels=class_dist.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Damage Type Distribution")

        # Timeline chart
        plt.subplot(1, 2, 2)
        detections = results.get("detections", [])
        if detections:
            timestamps = [d["timestamp"] for d in detections]
            damage_types = [d["class"] for d in detections]

            # Create scatter plot
            unique_types = list(set(damage_types))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))

            for i, damage_type in enumerate(unique_types):
                type_timestamps = [
                    t for t, dt in zip(timestamps, damage_types) if dt == damage_type
                ]
                plt.scatter(
                    type_timestamps,
                    [i] * len(type_timestamps),
                    c=[colors[i]],
                    label=damage_type,
                    s=100,
                    alpha=0.7,
                )

            plt.ylabel("Damage Type")
            plt.xlabel("Time (seconds)")
            plt.title("Detection Timeline")
            plt.yticks(range(len(unique_types)), unique_types)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close()

        return Image.open(img_buffer)

    def extract_youtube_video_id(self, url):
        """Extract video ID from YouTube URL"""
        import re

        # Patterns to match different YouTube URL formats
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([^&\n?#]+)",
            r"youtube\.com/watch\?.*v=([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def is_youtube_url(self, url: str) -> bool:
        """Check if the input is a YouTube URL"""
        youtube_indicators = [
            "youtube.com",
            "youtu.be",
            "youtube-nocookie.com",
            "www.youtube.com",
            "m.youtube.com",
        ]
        return any(indicator in url.lower() for indicator in youtube_indicators)

    def analyze_video_file(self, video_file, confidence, skip_frames):
        """Analyze uploaded video file"""
        try:
            if video_file is None:
                return None, None, "Please upload a video file", None, None, None

            self.logger.info(f"Processing uploaded video: {video_file.name}")

            # Extract base filename for video ID
            video_filename = os.path.basename(video_file.name)
            video_id = os.path.splitext(video_filename)[0]

            # Create organized output directory
            output_dir = self.create_organized_output_dir(video_id, "upload")

            # Create paths in organized structure
            videos_dir = os.path.join(output_dir, "videos")
            output_video = os.path.join(videos_dir, f"analyzed_{video_id}.mp4")

            # Analyze video
            results = self.detector.analyze_video(
                video_file.name,
                output_path=output_video,
                confidence=confidence / 100.0,  # Convert percentage to decimal
                skip_frames=skip_frames,
            )

            # Extract and save detection frames
            original_detections_count = len(results.get("detections", []))
            detection_images, saved_frame_paths = self.save_detection_frames_to_disk(
                video_file.name, results.get("detections", []), output_dir
            )

            # Log deduplication results
            if original_detections_count > 0:
                deduplicated_count = len(
                    self._deduplicate_detections(results.get("detections", []))
                )
                self.logger.info(
                    f"Deduplication: {original_detections_count} total detections -> {deduplicated_count} unique locations"
                )

            # Generate HTML report with saved frame paths
            html_report = self.generate_html_report_with_frames(
                results, "Uploaded Video", saved_frame_paths, output_dir
            )

            # Save HTML report to organized directory
            reports_dir = os.path.join(output_dir, "reports")
            report_path = os.path.join(
                reports_dir, f"video_damage_report_{video_id}.html"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            # Save analysis results JSON
            data_dir = os.path.join(output_dir, "data")
            results_json_path = os.path.join(
                data_dir, f"analysis_results_{video_id}.json"
            )
            with open(results_json_path, "w", encoding="utf-8") as f:
                # Make results JSON serializable
                json_results = {
                    "video_filename": video_filename,
                    "video_path": video_file.name,
                    "total_detections": results.get("total_detections", 0),
                    "total_frames": results.get("total_frames", 0),
                    "processing_time": results.get("processing_time", 0),
                    "class_distribution": results.get("class_distribution", {}),
                    "detections": results.get("detections", []),
                    "analysis_date": datetime.now().isoformat(),
                    "saved_frames": saved_frame_paths,
                }
                json.dump(json_results, f, indent=2)

            # Create results dataframe
            results_df = self.create_results_dataframe(results)

            # Create summary stats
            summary = self.create_summary_text(results)
            summary += f"\n\n📁 Results saved to: {output_dir}"

            # Add deduplication info to summary for video file analysis
            original_detections_count = len(results.get("detections", []))
            if original_detections_count > 0:
                deduplicated_count = len(
                    self._deduplicate_detections(results.get("detections", []))
                )
                summary += f"\n🔍 Original detections: {original_detections_count}"
                summary += f"\n🎯 Unique locations: {deduplicated_count}"
                summary += f"\n🖼️  Detection frames saved: {len(saved_frame_paths)}"
                if deduplicated_count < original_detections_count:
                    summary += f"\n✨ Removed {original_detections_count - deduplicated_count} duplicate detections from similar locations"
            else:
                summary += f"\n🖼️  Detection frames saved: {len(saved_frame_paths)}"

            # Create charts
            chart_image = self.create_detection_chart(results)

            return (
                output_video,
                results_df,
                summary,
                report_path,
                detection_images,
                chart_image,
            )

        except Exception as e:
            error_msg = f"Error analyzing video file: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return None, None, error_msg, None, None, None

    def analyze_youtube_video(self, youtube_url, confidence, stream_mode):
        """Analyze YouTube video"""
        try:
            if not youtube_url or not youtube_url.strip():
                return None, None, "Please enter a YouTube URL", None, None, None

            youtube_url = youtube_url.strip()

            if not self.is_youtube_url(youtube_url):
                return None, None, "Please enter a valid YouTube URL", None, None, None

            self.logger.info(f"Processing YouTube video: {youtube_url}")

            # Extract video ID for filename
            video_id = self.extract_youtube_video_id(youtube_url)
            if not video_id:
                video_id = "unknown_video"

            # Create organized output directory
            output_dir = self.create_organized_output_dir(video_id, "youtube")

            # Create temporary directory for processing
            temp_processing_dir = os.path.join(self.temp_dir, f"youtube_{video_id}")
            os.makedirs(temp_processing_dir, exist_ok=True)

            # Analyze video
            results = self.detector.analyze_youtube_video(
                youtube_url,
                output_dir=temp_processing_dir,
                confidence=confidence / 100.0,  # Convert percentage to decimal
                stream_only=stream_mode,
            )

            # Move analyzed video to organized directory if it exists
            output_video = None
            videos_dir = os.path.join(output_dir, "videos")

            if not stream_mode:
                for file in os.listdir(temp_processing_dir):
                    if file.startswith("analyzed_") and file.endswith(".mp4"):
                        source_path = os.path.join(temp_processing_dir, file)
                        output_video = os.path.join(
                            videos_dir, f"analyzed_{video_id}.mp4"
                        )
                        shutil.move(source_path, output_video)
                        break

            # Extract and save detection frames
            detection_images = []
            saved_frame_paths = []
            original_detections_count = len(results.get("detections", []))

            if not stream_mode and output_video and os.path.exists(output_video):
                # Use processed video for frame extraction
                detection_images, saved_frame_paths = (
                    self.save_detection_frames_to_disk(
                        output_video, results.get("detections", []), output_dir
                    )
                )
            elif stream_mode and results.get("detections"):
                # In streaming mode, extract frames directly from YouTube URL
                try:
                    detection_images, saved_frame_paths = (
                        self.extract_and_save_frames_from_youtube(
                            youtube_url, results.get("detections", []), output_dir
                        )
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract detection frames from YouTube: {e}"
                    )
                    detection_images = []
                    saved_frame_paths = []

            # Log deduplication results for YouTube
            if original_detections_count > 0:
                deduplicated_count = len(
                    self._deduplicate_detections(results.get("detections", []))
                )
                self.logger.info(
                    f"YouTube Deduplication: {original_detections_count} total detections -> {deduplicated_count} unique locations"
                )

            # Generate HTML report with saved frame paths
            html_report = self.generate_html_report_with_frames(
                results,
                f"YouTube: {results.get('video_title', 'Unknown')}",
                saved_frame_paths,
                output_dir,
            )

            # Save HTML report to organized directory
            reports_dir = os.path.join(output_dir, "reports")
            report_path = os.path.join(
                reports_dir, f"youtube_damage_report_{video_id}.html"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            # Save analysis results JSON
            data_dir = os.path.join(output_dir, "data")
            results_json_path = os.path.join(
                data_dir, f"analysis_results_{video_id}.json"
            )
            with open(results_json_path, "w", encoding="utf-8") as f:
                # Make results JSON serializable
                json_results = {
                    "video_title": results.get("video_title", "Unknown"),
                    "video_url": youtube_url,
                    "total_detections": results.get("total_detections", 0),
                    "total_frames": results.get("total_frames", 0),
                    "processing_time": results.get("processing_time", 0),
                    "class_distribution": results.get("class_distribution", {}),
                    "detections": results.get("detections", []),
                    "analysis_date": datetime.now().isoformat(),
                    "saved_frames": saved_frame_paths,
                }
                json.dump(json_results, f, indent=2)

            # Create results dataframe
            results_df = self.create_results_dataframe(results)

            # Create summary stats
            summary = self.create_summary_text(results)
            summary += f"\n\n📁 Results saved to: {output_dir}"

            # Add deduplication info to summary for YouTube analysis
            original_detections_count = len(results.get("detections", []))
            if original_detections_count > 0:
                deduplicated_count = len(
                    self._deduplicate_detections(results.get("detections", []))
                )
                summary += f"\n🔍 Original detections: {original_detections_count}"
                summary += f"\n🎯 Unique locations: {deduplicated_count}"
                summary += f"\n🖼️  Detection frames saved: {len(saved_frame_paths)}"
                if deduplicated_count < original_detections_count:
                    summary += f"\n✨ Removed {original_detections_count - deduplicated_count} duplicate detections from similar locations"
            else:
                summary += f"\n🖼️  Detection frames saved: {len(saved_frame_paths)}"

            # Create charts
            chart_image = self.create_detection_chart(results)

            return (
                output_video,
                results_df,
                summary,
                report_path,
                detection_images,
                chart_image,
            )

        except Exception as e:
            error_msg = f"Error analyzing YouTube video: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return None, None, error_msg, None, None, None

    def create_results_dataframe(self, results):
        """Create a pandas dataframe from detection results"""
        detections = results.get("detections", [])

        if not detections:
            # Return empty dataframe with expected columns
            return pd.DataFrame(
                columns=["Frame", "Timestamp (s)", "Damage Type", "Confidence", "Bbox"]
            )

        # Convert detections to dataframe
        data = []
        for detection in detections:
            data.append(
                {
                    "Frame": detection.get("frame", "N/A"),
                    "Timestamp (s)": f"{detection.get('timestamp', 0):.2f}",
                    "Damage Type": detection.get("class", "Unknown"),
                    "Confidence": f"{detection.get('confidence', 0):.3f}",
                    "Bbox": f"[{detection.get('bbox', [])}]",
                }
            )

        df = pd.DataFrame(data)
        return df.head(100)  # Limit to first 100 for display

    def create_summary_text(self, results):
        """Create summary text from results"""
        total_detections = results.get("total_detections", 0)
        duration = results.get("video_duration", 0)
        damage_density = results.get("damage_density", 0)

        summary = f"""
## 📊 Analysis Summary

**🎯 Total Damage Detections:** {total_detections}
**⏱️ Video Duration:** {duration:.1f} seconds
**📈 Damage Density:** {damage_density:.2f} damages per second

"""

        # Add class distribution if available
        class_dist = results.get("class_distribution", {})
        if class_dist:
            summary += "### 🔍 Damage Types Detected:\n"
            for damage_type, count in sorted(
                class_dist.items(), key=lambda x: x[1], reverse=True
            ):
                summary += f"- **{damage_type}:** {count} detections\n"

        # Add processing info
        processed_frames = results.get("processed_frames", "N/A")
        total_frames = results.get("total_frames", "N/A")

        summary += f"""
### 📱 Processing Details:
- **Frames Processed:** {processed_frames}
- **Total Frames:** {total_frames}
"""

        if results.get("streaming"):
            summary += "- **Analysis Method:** Streaming (no download)\n"
        else:
            summary += "- **Analysis Method:** Full download and analysis\n"

        return summary

    def generate_html_report(self, results, video_title):
        """Generate HTML report similar to existing format but enhanced"""
        detections = results.get("detections", [])
        total_detections = results.get("total_detections", 0)
        duration = results.get("video_duration", 0)
        damage_density = results.get("damage_density", 0)
        class_dist = results.get("class_distribution", {})

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Road Damage Detection Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .stats {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0; 
        }}
        .stat-box {{ 
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 25px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(116, 185, 255, 0.3);
        }}
        .stat-box h3 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        .stat-box p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .damage-types {{
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
            border-left: 5px solid #28a745;
        }}
        .damage-types h2 {{
            color: #28a745;
            margin-top: 0;
        }}
        .damage-type-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .damage-type-item:last-child {{
            border-bottom: none;
        }}
        .damage-type-name {{
            font-weight: 600;
            color: #495057;
        }}
        .damage-type-count {{
            background-color: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .detections {{ 
            margin-top: 30px; 
        }}
        .detections h2 {{
            color: #495057;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px;
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{ 
            background: linear-gradient(135deg, #495057 0%, #6c757d 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{ 
            border-bottom: 1px solid #dee2e6; 
            padding: 12px 15px; 
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        .confidence-bar {{
            display: inline-block;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-left: 10px;
            width: 100px;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
        }}
        .damage-type-tag {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: white;
        }}
        .longitudinal {{ background-color: #dc3545; }}
        .transverse {{ background-color: #fd7e14; }}
        .alligator {{ background-color: #6f42c1; }}
        .potholes {{ background-color: #20c997; }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            color: #6c757d;
        }}
        .processing-info {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #2196f3;
        }}
        .processing-info h3 {{
            color: #1976d2;
            margin-top: 0;
        }}
        .no-detections {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .no-detections h3 {{
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛣️ Road Damage Detection Report</h1>
            <p><strong>Video:</strong> {video_title}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>{total_detections}</h3>
                <p>Total Damage Detections</p>
            </div>
            <div class="stat-box">
                <h3>{duration:.1f}s</h3>
                <p>Video Duration</p>
            </div>
            <div class="stat-box">
                <h3>{damage_density:.2f}</h3>
                <p>Damages per Second</p>
            </div>
        </div>
        
        <div class="processing-info">
            <h3>📊 Processing Information</h3>
            <p><strong>Total Frames:</strong> {results.get('total_frames', 'N/A')}</p>
            <p><strong>Processed Frames:</strong> {results.get('processed_frames', 'N/A')}</p>
            <p><strong>Analysis Method:</strong> {'Streaming' if results.get('streaming') else 'Full Download'}</p>
        </div>
"""

        # Add damage types distribution
        if class_dist:
            html_content += """
        <div class="damage-types">
            <h2>🔍 Detected Damage Types</h2>
"""
            for damage_type, count in sorted(
                class_dist.items(), key=lambda x: x[1], reverse=True
            ):
                html_content += f"""
            <div class="damage-type-item">
                <span class="damage-type-name">{damage_type}</span>
                <span class="damage-type-count">{count}</span>
            </div>
"""
            html_content += """
        </div>
"""

        # Add detections table
        html_content += """
        <div class="detections">
            <h2>🎯 Detection Details</h2>
"""

        if detections:
            html_content += """
            <table>
                <tr>
                    <th>Frame</th>
                    <th>Timestamp</th>
                    <th>Damage Type</th>
                    <th>Confidence</th>
                    <th>Bounding Box</th>
                </tr>
"""

            # Show first 100 detections
            for detection in detections[:100]:
                damage_type = detection.get("class", "unknown")
                confidence = detection.get("confidence", 0)

                # Get CSS class for damage type
                damage_class = damage_type.lower().replace(" ", "").replace("crack", "")
                if "longitudinal" in damage_class:
                    damage_class = "longitudinal"
                elif "transverse" in damage_class:
                    damage_class = "transverse"
                elif "alligator" in damage_class:
                    damage_class = "alligator"
                elif "pothole" in damage_class:
                    damage_class = "potholes"

                bbox = detection.get("bbox", [])
                bbox_str = f"[{', '.join(map(str, bbox))}]" if bbox else "N/A"

                html_content += f"""
                <tr>
                    <td>{detection.get('frame', 'N/A')}</td>
                    <td>{detection.get('timestamp', 0):.2f}s</td>
                    <td>
                        <span class="damage-type-tag {damage_class}">{damage_type}</span>
                    </td>
                    <td>
                        {confidence:.3f}
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%"></div>
                        </div>
                    </td>
                    <td style="font-family: monospace; font-size: 0.9em;">{bbox_str}</td>
                </tr>
"""

            if len(detections) > 100:
                html_content += f"""
                <tr>
                    <td colspan="5" style="text-align: center; font-style: italic; color: #6c757d;">
                        ... and {len(detections) - 100} more detections (showing first 100)
                    </td>
                </tr>
"""

            html_content += """
            </table>
"""
        else:
            html_content += """
            <div class="no-detections">
                <h3>No Damage Detected</h3>
                <p>The analysis did not find any road damage in this video.</p>
                <p>This could mean:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>The road conditions are good</li>
                    <li>The confidence threshold might be too high</li>
                    <li>The lighting or video quality affects detection</li>
                </ul>
            </div>
"""

        html_content += """
        </div>
        
        <div class="footer">
            <p>Generated by Road Damage Detection System using YOLOv8</p>
            <p>🤖 Powered by Ultralytics YOLO | 🚀 Built with Gradio</p>
        </div>
    </div>
</body>
</html>
"""

        return html_content

    def generate_html_report_with_frames(
        self, results, video_title, saved_frame_paths, output_dir
    ):
        """Generate enhanced HTML report with saved frame images"""
        detections = results.get("detections", [])
        total_detections = results.get("total_detections", 0)
        duration = results.get("video_duration", 0)
        damage_density = results.get("damage_density", 0)
        class_dist = results.get("class_distribution", {})

        # Create relative paths for frames in HTML
        frames_html = ""
        if saved_frame_paths:
            frames_html = """
            <div class="detection-frames">
                <h2>🖼️ Detection Frames</h2>
                <p>Detected damage instances with bounding boxes and timestamps:</p>
                <div class="frame-gallery">
            """
            for i, frame_path in enumerate(saved_frame_paths):
                # Get relative path from report to frame
                rel_path = os.path.relpath(
                    frame_path, os.path.join(output_dir, "reports")
                )
                frame_filename = os.path.basename(frame_path)

                # Extract info from filename
                parts = (
                    frame_filename.replace("detection_", "")
                    .replace(".jpg", "")
                    .split("_")
                )
                if len(parts) >= 4:
                    frame_num = parts[1] + "_" + parts[2]
                    damage_type = "_".join(parts[3:])
                else:
                    frame_num = f"frame_{i+1}"
                    damage_type = "damage"

                frames_html += f"""
                    <div class="frame-item">
                        <img src="{rel_path}" alt="Detection {i+1}" class="detection-image">
                        <div class="frame-info">
                            <p><strong>Frame:</strong> {frame_num}</p>
                            <p><strong>Type:</strong> {damage_type.replace('_', ' ').title()}</p>
                        </div>
                    </div>
                """
            frames_html += """
                </div>
            </div>
            """

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Road Damage Detection Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .stats {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0; 
        }}
        .stat-box {{ 
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 25px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(116, 185, 255, 0.3);
        }}
        .stat-box h3 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        .stat-box p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .damage-types {{
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
            border-left: 5px solid #28a745;
        }}
        .damage-types h2 {{
            color: #28a745;
            margin-top: 0;
        }}
        .damage-type-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .damage-type-item:last-child {{
            border-bottom: none;
        }}
        .damage-type-name {{
            font-weight: 600;
            color: #495057;
        }}
        .damage-type-count {{
            background-color: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
        }}
        .detection-frames {{
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
            border-left: 5px solid #e74c3c;
        }}
        .detection-frames h2 {{
            color: #e74c3c;
            margin-top: 0;
        }}
        .frame-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .frame-item {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .detection-image {{
            width: 100%;
            max-width: 400px;
            height: auto;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .frame-info {{
            text-align: left;
        }}
        .frame-info p {{
            margin: 5px 0;
            color: #495057;
        }}
        .detections-table {{
            margin: 30px 0;
            overflow-x: auto;
        }}
        .detections-table table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .detections-table th {{
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .detections-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .detections-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .detections-table tr:hover {{
            background-color: #e9ecef;
        }}
        .confidence-high {{ color: #28a745; font-weight: bold; }}
        .confidence-medium {{ color: #ffc107; font-weight: bold; }}
        .confidence-low {{ color: #dc3545; font-weight: bold; }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            color: #6c757d;
        }}
        .footer p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛣️ Road Damage Detection Report</h1>
            <p>{video_title}</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="stats">
            <div class="stat-box">
                <h3>{total_detections}</h3>
                <p>Total Detections</p>
            </div>
            <div class="stat-box">
                <h3>{duration:.1f}s</h3>
                <p>Video Duration</p>
            </div>
            <div class="stat-box">
                <h3>{damage_density:.2f}</h3>
                <p>Damages per Minute</p>
            </div>
            <div class="stat-box">
                <h3>{len(saved_frame_paths)}</h3>
                <p>Saved Detection Frames</p>
            </div>
        </div>

        {frames_html}

        <div class="damage-types">
            <h2>🔍 Damage Type Distribution</h2>
        """

        # Add damage type distribution
        for damage_type, count in class_dist.items():
            html_content += f"""
            <div class="damage-type-item">
                <span class="damage-type-name">{damage_type.replace('_', ' ').title()}</span>
                <span class="damage-type-count">{count}</span>
            </div>
            """

        html_content += """
        </div>

        <div class="detections-table">
            <h2>📋 Detection Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Time</th>
                        <th>Frame</th>
                        <th>Damage Type</th>
                        <th>Confidence</th>
                        <th>Location (x, y, w, h)</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Add detection details
        for i, detection in enumerate(detections, 1):
            confidence = detection["confidence"]
            confidence_class = (
                "confidence-high"
                if confidence > 0.8
                else "confidence-medium"
                if confidence > 0.5
                else "confidence-low"
            )

            bbox = detection["bbox"]
            location = f"({bbox[0]}, {bbox[1]}, {bbox[2]-bbox[0]}, {bbox[3]-bbox[1]})"

            html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{self.format_timestamp(detection['timestamp'])}</td>
                        <td>{detection['frame']}</td>
                        <td>{detection['class'].replace('_', ' ').title()}</td>
                        <td><span class="{confidence_class}">{confidence:.2%}</span></td>
                        <td>{location}</td>
                    </tr>
            """

        html_content += f"""
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p><strong>📁 Analysis Results Location:</strong> {output_dir}</p>
            <p>Generated by Road Damage Detection System using YOLOv8</p>
            <p>🤖 Powered by Ultralytics YOLO | 🚀 Built with Gradio</p>
        </div>
    </div>
</body>
</html>
"""

        return html_content

    def create_interface(self):
        """Create the Gradio interface"""

        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """

        with gr.Blocks(css=custom_css, title="Road Damage Detection") as interface:
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🛣️ Road Damage Detection System</h1>
                <p>Upload a video file or provide a YouTube link to analyze road damage using AI</p>
            </div>
            """)

            with gr.Tabs():
                # Tab 1: Video Upload
                with gr.TabItem("📁 Upload Video"):
                    gr.Markdown("""
                    ### Upload a video file for analysis
                    Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V (case insensitive)
                    """)

                    with gr.Row():
                        with gr.Column(scale=2):
                            video_input = gr.File(
                                label="Select Video File",
                                file_types=[
                                    ".mp4",
                                    ".avi",
                                    ".mov",
                                    ".mkv",
                                    ".wmv",
                                    ".flv",
                                    ".webm",
                                    ".m4v",
                                    ".MP4",
                                    ".AVI",
                                    ".MOV",
                                    ".MKV",
                                    ".WMV",
                                    ".FLV",
                                    ".WEBM",
                                    ".M4V",
                                ],
                                type="filepath",
                            )

                            with gr.Row():
                                confidence_slider = gr.Slider(
                                    minimum=10,
                                    maximum=90,
                                    value=30,
                                    step=5,
                                    label="Confidence Threshold (%)",
                                    info="Lower values detect more potential damage (may include false positives)",
                                )

                                skip_frames_slider = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    label="Frame Skip",
                                    info="Process every Nth frame (higher = faster but less detailed)",
                                )

                            analyze_btn = gr.Button(
                                "🔍 Analyze Video", variant="primary", size="lg"
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("""
                            #### 💡 Tips:
                            - **Confidence Threshold**: Lower values (20-40%) detect more damage but may include false positives
                            - **Frame Skip**: Higher values process faster but might miss brief damage appearances
                            - **Best Results**: Use good quality videos with clear road visibility
                            """)

                    # Results section for video upload
                    with gr.Row():
                        with gr.Column(scale=2):
                            video_summary = gr.Markdown(label="Analysis Summary")
                            video_results_table = gr.Dataframe(
                                label="Detection Results", interactive=False, wrap=True
                            )

                        with gr.Column(scale=2):
                            output_video = gr.Video(
                                label="Analyzed Video (with annotations)"
                            )

                    with gr.Row():
                        with gr.Column():
                            detection_images = gr.Gallery(
                                label="🎯 Detection Images (Top Detections)",
                                show_label=True,
                                columns=3,
                                height="auto",
                            )

                        with gr.Column():
                            analysis_chart = gr.Image(label="📊 Analysis Charts")

                    video_report_file = gr.File(
                        label="📋 Download HTML Report", visible=False
                    )

                    # Connect video analysis
                    analyze_btn.click(
                        fn=self.analyze_video_file,
                        inputs=[video_input, confidence_slider, skip_frames_slider],
                        outputs=[
                            output_video,
                            video_results_table,
                            video_summary,
                            video_report_file,
                            detection_images,
                            analysis_chart,
                        ],
                        show_progress=True,
                    )

                # Tab 2: YouTube Analysis
                with gr.TabItem("📺 YouTube Video"):
                    gr.Markdown("""
                    ### Analyze YouTube videos for road damage
                    Enter a YouTube URL to analyze road conditions
                    """)

                    with gr.Row():
                        with gr.Column(scale=2):
                            youtube_input = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://youtube.com/watch?v=...",
                                info="Paste YouTube video URL here",
                            )

                            with gr.Row():
                                yt_confidence_slider = gr.Slider(
                                    minimum=10,
                                    maximum=90,
                                    value=30,
                                    step=5,
                                    label="Confidence Threshold (%)",
                                )

                                stream_mode = gr.Checkbox(
                                    label="Stream Mode",
                                    value=True,
                                    info="Stream without downloading (faster, no output video)",
                                )

                            analyze_yt_btn = gr.Button(
                                "🔍 Analyze YouTube Video", variant="primary", size="lg"
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("""
                            #### 🎥 YouTube Analysis:
                            - **Stream Mode**: Faster analysis, no download needed, no output video
                            - **Download Mode**: Downloads video, creates annotated output (slower)
                            - Works with most public YouTube videos
                            - Some videos may be restricted
                            """)

                    # Results section for YouTube
                    with gr.Row():
                        with gr.Column(scale=2):
                            yt_summary = gr.Markdown(label="Analysis Summary")
                            yt_results_table = gr.Dataframe(
                                label="Detection Results", interactive=False, wrap=True
                            )

                        with gr.Column(scale=2):
                            yt_output_video = gr.Video(
                                label="Analyzed Video (if downloaded)"
                            )

                    with gr.Row():
                        with gr.Column():
                            yt_detection_images = gr.Gallery(
                                label="🎯 Detection Images (if available)",
                                show_label=True,
                                columns=3,
                                height="auto",
                            )

                        with gr.Column():
                            yt_analysis_chart = gr.Image(label="📊 Analysis Charts")

                    yt_report_file = gr.File(
                        label="📋 Download HTML Report", visible=False
                    )

                    # Connect YouTube analysis
                    analyze_yt_btn.click(
                        fn=self.analyze_youtube_video,
                        inputs=[youtube_input, yt_confidence_slider, stream_mode],
                        outputs=[
                            yt_output_video,
                            yt_results_table,
                            yt_summary,
                            yt_report_file,
                            yt_detection_images,
                            yt_analysis_chart,
                        ],
                        show_progress=True,
                    )

                # Tab 3: About
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ## 🛣️ Road Damage Detection System
                    
                    This application uses **YOLOv8** machine learning model to detect and classify road damage in videos.
                    
                    ### 🎯 Detected Damage Types:
                    - **Longitudinal Crack**: Cracks running parallel to road direction
                    - **Transverse Crack**: Cracks running perpendicular to road direction  
                    - **Alligator Crack**: Interconnected cracks resembling alligator skin
                    - **Potholes**: Holes or depressions in road surface
                    
                    ### 🔧 How it Works:
                    1. **Upload** a video or provide a **YouTube link**
                    2. The system processes each frame using **AI detection**
                    3. **Damage locations** are identified with confidence scores
                    4. Results include **timestamps**, **damage types**, and **bounding boxes**
                    5. Generate **detailed HTML reports** for documentation
                    
                    ### 📊 Output Information:
                    - **Frame number** and **timestamp** of each detection
                    - **Damage classification** with confidence percentage
                    - **Bounding box coordinates** for precise location
                    - **Summary statistics** and damage density analysis
                    - **Annotated video** with highlighted damage areas (when available)
                    
                    ### 💡 Tips for Best Results:
                    - Use **high-quality videos** with clear road visibility
                    - Ensure **good lighting** conditions
                    - **Stable camera** movement works better than shaky footage
                    - Videos with **perpendicular view** of road surface are ideal
                    
                    ### 🚀 Technology Stack:
                    - **YOLOv8** for object detection
                    - **OpenCV** for video processing  
                    - **Gradio** for web interface
                    - **PyTubeFix** for YouTube video handling
                    
                    ---
                    *Built with ❤️ using cutting-edge AI technology*
                    """)

        return interface

    def launch(self, **kwargs):
        """Launch the Gradio application"""
        interface = self.create_interface()
        return interface.launch(**kwargs)


def main():
    """Main function to run the application"""
    try:
        # Create and launch the app
        app = RoadDamageGradioApp()

        # Launch with custom settings
        app.launch(
            share=True,  # Create public link for accessibility
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,  # Default Gradio port
            show_error=True,
            favicon_path=None,
            ssl_verify=False,
            inbrowser=True,  # Open browser automatically
        )

    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
