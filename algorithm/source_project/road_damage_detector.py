import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadDamageDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Road Damage Detector

        Args:
            model_path: Path to custom trained YOLO model. If None, uses trained RDD model
        """
        # Default to using the trained RDD model
        if model_path is None:
            # Look for the trained model in the expected location
            trained_model_paths = [
                "runs/RDD_Training/test_training/weights/best.pt",
                "runs/RDD_Training/test_training2/weights/best.pt",
                "runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt",
                "models/best.pt",
            ]

            for path in trained_model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                logger.warning("No trained RDD model found, using YOLOv8n")
                model_path = "yolov8n.pt"

        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Fallback
            logger.info("Loaded YOLOv8n model")

        # Road damage class names (matching RDD2022 dataset)
        self.damage_classes = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]


    def detect_frame(
        self, frame: np.ndarray, confidence: float = 0.5
    ) -> Tuple[np.ndarray, list]:
        """
        Detect road damage in a single frame

        Args:
            frame: Input frame
            confidence: Confidence threshold

        Returns:
            Annotated frame and detection results
        """
        results = self.model(frame, conf=confidence)

        detections = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = (
                        self.model.names[cls]
                        if cls < len(self.model.names)
                        else "damage"
                    )

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(conf),
                            "class": class_name,
                            "class_id": cls,
                        }
                    )

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )

                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return annotated_frame, detections

    def analyze_video(
            self,
            video_path: str,
            output_path: str = None,
            confidence: float = 0.5,
            skip_frames: int = 5,
            progress_callback=None,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames"
        )
        print(
            f"[detector] start analyze_video: video_path={video_path}, "
            f"output_path={output_path}, confidence={confidence}, skip_frames={skip_frames}"
        )

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[detector] video writer created: {output_path}")

        all_detections = []
        damage_count = 0
        processed_frames = 0
        frame_count = 0
        class_counts = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[detector] cap.read() returned False, stop")
                    break

                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue

                annotated_frame, detections = self.detect_frame(frame, confidence)

                current_frame_detections = []

                if detections:
                    damage_count += len(detections)

                    for detection in detections:
                        detection["frame"] = frame_count
                        detection["timestamp"] = frame_count / fps

                        class_name = detection["class"]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        current_frame_detections.append(
                            {
                                "bbox": detection["bbox"],
                                "confidence": float(detection["confidence"]),
                                "class_name": class_name,
                                "class_id": detection["class_id"],
                            }
                        )

                    all_detections.extend(detections)

                if out is not None:
                    out.write(annotated_frame)

                processed_frames += 1
                progress = int((frame_count + 1) / total_frames * 100) if total_frames > 0 else 0

                if progress_callback:
                    frame_bytes = b""
                    frame_base64 = ""

                    ok, buffer = cv2.imencode(".jpg", annotated_frame)
                    print(
                        f"[detector] frame={frame_count}, processed={processed_frames}, "
                        f"progress={progress}, encode_ok={ok}, detections={len(current_frame_detections)}"
                    )

                    if ok:
                        frame_bytes = buffer.tobytes()
                        frame_base64 = (
                                "data:image/jpeg;base64,"
                                + base64.b64encode(frame_bytes).decode("utf-8")
                        )
                        print(
                            f"[detector] frame={frame_count}, frame_bytes_len={len(frame_bytes)}, "
                            f"base64_len={len(frame_base64)}"
                        )
                    else:
                        print(f"[detector] frame={frame_count}, jpeg encode failed")

                    payload = {
                        "type": "progress",
                        "progress": progress,
                        "current_frame": frame_count,
                        "total_frames": total_frames,
                        "detections": current_frame_detections,
                        "statistics": class_counts.copy(),
                        "frame_image": frame_base64,
                        "frame_bytes": frame_bytes,
                    }

                    print(
                        f"[detector] callback payload: "
                        f"has_frame_bytes={bool(frame_bytes)}, "
                        f"stats_keys={list(class_counts.keys())}"
                    )

                    progress_callback(payload)

                if processed_frames % 50 == 0:
                    logger.info(
                        f"Progress: {progress:.1f}% - Detected {damage_count} damages"
                    )

                frame_count += 1

        finally:
            cap.release()
            if out is not None:
                out.release()
            print("[detector] resources released")

        results = {
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_detections": damage_count,
            "detections": all_detections,
            "video_duration": total_frames / fps if fps > 0 else 0,
            "damage_density": damage_count / (total_frames / fps) if fps > 0 and total_frames > 0 else 0,
        }

        if all_detections:
            results["class_distribution"] = class_counts
            results["most_common_damage"] = max(class_counts, key=class_counts.get)

        print(
            f"[detector] finished: processed_frames={processed_frames}, "
            f"total_detections={damage_count}, class_counts={class_counts}"
        )

        return results


    def analyze_youtube_video(
        self,
        youtube_url: str,
        output_dir: str = "output",
        confidence: float = 0.5,
        stream_only: bool = False,
    ) -> dict:
        """
        Analyze YouTube video for road damage with automatic fallback

        Args:
            youtube_url: YouTube video URL
            output_dir: Directory to save results
            confidence: Detection confidence threshold
            stream_only: If True, only stream analysis is performed

        Returns:
            Analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if stream_only:
            # Use streaming method (no download, no output video)
            logger.info("Using streaming mode (no download required)")
            results = self.analyze_youtube_video_stream(youtube_url, confidence)

            # Save results to output directory
            results_path = os.path.join(output_dir, "analysis_results.json")

            import json

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Stream analysis complete. Results saved to {results_path}")
            return results

        else:
            # Try download method first, fallback to streaming if download fails
            try:
                logger.info("Attempting to download YouTube video...")
                video_path = self.download_youtube_video(youtube_url, output_dir)

                # Analyze video
                logger.info("Analyzing downloaded video for road damage...")
                output_video_path = os.path.join(
                    output_dir, "analyzed_" + os.path.basename(video_path)
                )
                results = self.analyze_video(video_path, output_video_path, confidence)

                # Save results
                results_path = os.path.join(output_dir, "analysis_results.json")
                import json

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                logger.info(f"Analysis complete. Results saved to {results_path}")
                logger.info(f"Annotated video saved to {output_video_path}")

                return results

            except Exception as download_error:
                logger.warning(f"Download failed: {download_error}")
                logger.info("Falling back to streaming mode...")

                try:
                    # Fallback to streaming
                    results = self.analyze_youtube_video_stream(youtube_url, confidence)

                    # Save results to output directory
                    results_path = os.path.join(output_dir, "analysis_results.json")

                    import json

                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                    logger.info(
                        f"Fallback streaming analysis complete. Results saved to {results_path}"
                    )
                    logger.info("Note: No output video generated (streaming mode)")

                    return results

                except Exception as stream_error:
                    logger.error(f"Both download and streaming failed!")
                    logger.error(f"Download error: {download_error}")
                    logger.error(f"Streaming error: {stream_error}")
                    raise Exception(
                        f"Cannot analyze YouTube video. Download failed: {download_error}, Streaming failed: {stream_error}"
                    )

    def generate_report(self, results: dict, output_path: str = "damage_report.html"):
        """
        Generate HTML report from analysis results
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Road Damage Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #e9e9e9; padding: 15px; border-radius: 5px; text-align: center; }}
                .detections {{ margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Road Damage Detection Report</h1>
                <p><strong>Video:</strong> {results.get('video_path', 'N/A')}</p>
                <p><strong>Analysis Date:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{results.get('total_detections', 0)}</h3>
                    <p>Total Damage Detections</p>
                </div>
                <div class="stat-box">
                    <h3>{results.get('video_duration', 0):.1f}s</h3>
                    <p>Video Duration</p>
                </div>
                <div class="stat-box">
                    <h3>{results.get('damage_density', 0):.2f}</h3>
                    <p>Damages per Second</p>
                </div>
            </div>
            
            <div class="detections">
                <h2>Detection Summary</h2>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Timestamp (s)</th>
                        <th>Damage Type</th>
                        <th>Confidence</th>
                    </tr>
        """

        for detection in results.get("detections", [])[:50]:  # Show first 50 detections
            html_content += f"""
                    <tr>
                        <td>{detection.get('frame', 'N/A')}</td>
                        <td>{detection.get('timestamp', 0):.2f}</td>
                        <td>{detection.get('class', 'unknown')}</td>
                        <td>{detection.get('confidence', 0):.2f}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Report generated: {output_path}")


def main():
    """Example usage"""
    # Initialize detector
    detector = RoadDamageDetector()

    # Example 1: Analyze local video
    local_video = "/home/bagus/github/road-damage-detection/15-01-2024/C0007.MP4"
    if os.path.exists(local_video):
        print("Analyzing local video...")
        results = detector.analyze_video(
            local_video, output_path="output/analyzed_local_video.mp4", confidence=0.3
        )
        detector.generate_report(results, "local_video_report.html")
        print(f"Found {results['total_detections']} potential road damages")

    # Example 2: Analyze YouTube video (uncomment to use)
    # youtube_url = "YOUR_YOUTUBE_URL_HERE"
    # results = detector.analyze_youtube_video(youtube_url, confidence=0.3)
    # detector.generate_report(results, "youtube_video_report.html")


if __name__ == "__main__":
    main()
