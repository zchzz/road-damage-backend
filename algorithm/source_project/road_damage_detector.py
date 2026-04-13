import base64
import logging
import os
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadDamageDetector:
    def __init__(self, model_path: str | None = None):
        """
        初始化道路病害检测器

        Args:
            model_path: YOLO 权重路径；为空时自动尝试项目内常见路径
        """
        if model_path is None:
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
                logger.warning("No trained RDD model found, using YOLOv8n fallback")
                model_path = "yolov8n.pt"

        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info("Loaded model from %s", model_path)
        else:
            self.model = YOLO("yolov8n.pt")
            logger.info("Loaded YOLOv8n fallback model")

        self.damage_classes = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]

    def detect_frame(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
    ) -> Tuple[np.ndarray, list]:
        """
        对单帧进行检测，返回标注后的图像和检测结果
        """
        results = self.model(frame, conf=confidence)

        detections = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                class_name = (
                    self.model.names[cls]
                    if cls < len(self.model.names)
                    else "damage"
                )

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "class": class_name,
                        "class_id": cls,
                    }
                )

                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                label = f"{class_name}: {conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), max(0, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return annotated_frame, detections

    def analyze_video(
        self,
        video_path: str,
        output_path: str | None = None,
        confidence: float = 0.5,
        skip_frames: int = 5,
        progress_callback=None,
    ) -> dict:
        """
        同步分析视频；如果传入 progress_callback，会在处理过程中持续回调
        """
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
            "Video properties: %sx%s, %s FPS, %s frames",
            width,
            height,
            fps,
            total_frames,
        )

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info("Video writer created: %s", output_path)

        all_detections = []
        damage_count = 0
        processed_frames = 0
        frame_count = 0
        class_counts: dict[str, int] = {}

        # 兜底，避免 skip_frames = 0
        if skip_frames <= 0:
            skip_frames = 1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video read finished")
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
                        detection["timestamp"] = frame_count / fps if fps > 0 else 0

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
                progress = (
                    int((frame_count + 1) / total_frames * 100)
                    if total_frames > 0
                    else 0
                )

                if progress_callback:
                    frame_bytes = b""
                    frame_base64 = ""

                    ok, buffer = cv2.imencode(".jpg", annotated_frame)
                    if ok:
                        frame_bytes = buffer.tobytes()
                        frame_base64 = (
                            "data:image/jpeg;base64,"
                            + base64.b64encode(frame_bytes).decode("utf-8")
                        )

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

                    progress_callback(payload)

                if processed_frames % 50 == 0:
                    logger.info(
                        "Progress: %s%% - Detected %s damages",
                        progress,
                        damage_count,
                    )

                frame_count += 1

        finally:
            cap.release()
            if out is not None:
                out.release()

        results = {
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_detections": damage_count,
            "detections": all_detections,
            "video_duration": total_frames / fps if fps > 0 else 0,
            "damage_density": (
                damage_count / (total_frames / fps)
                if fps > 0 and total_frames > 0
                else 0
            ),
        }

        if class_counts:
            results["class_distribution"] = class_counts
            results["most_common_damage"] = max(class_counts, key=class_counts.get)

        logger.info(
            "Finished: processed_frames=%s, total_detections=%s, class_counts=%s",
            processed_frames,
            damage_count,
            class_counts,
        )

        return results

    def generate_report(self, results: dict, output_path: str = "damage_report.html"):
        """
        生成 HTML 报告
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Road Damage Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; gap: 16px; }}
                .stat-box {{ background-color: #e9e9e9; padding: 15px; border-radius: 5px; text-align: center; flex: 1; }}
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
                <p><strong>Total Detections:</strong> {results.get('total_detections', 0)}</p>
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

        for detection in results.get("detections", [])[:50]:
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

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Report generated: %s", output_path)