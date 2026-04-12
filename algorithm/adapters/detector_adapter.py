from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PROJECT_DIR = BASE_DIR / "algorithm" / "source_project"
WEIGHTS_DIR = BASE_DIR / "algorithm" / "weights"

if str(SOURCE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_PROJECT_DIR))

from road_damage_detector import RoadDamageDetector


def _normalize_class_distribution(class_distribution: dict | None) -> dict:
    if not class_distribution:
        return {}
    return {str(k): int(v) for k, v in class_distribution.items()}


def detect_video(
    video_path: str,
    output_video_path: str,
    confidence: float = 0.3,
    skip_frames: int = 5,
    model_path: str | None = None,
    report_path: str | None = None,
    result_json_path: str | None = None,
    progress_callback=None,
):
    model_file = Path(model_path) if model_path else (WEIGHTS_DIR / "best.pt")

    if not model_file.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_file}")

    detector = RoadDamageDetector(str(model_file))

    results = detector.analyze_video(
        video_path=video_path,
        output_path=output_video_path,
        confidence=confidence,
        skip_frames=skip_frames,
        progress_callback=progress_callback,
    )

    if report_path:
        detector.generate_report(results, report_path)

    normalized = {
        "summary": {
            "total_detections": int(results.get("total_detections", 0)),
            "video_duration": float(results.get("video_duration", 0)),
            "processed_frames": int(results.get("processed_frames", 0)),
            "damage_density": float(results.get("damage_density", 0)),
            "damage_types": _normalize_class_distribution(
                results.get("class_distribution", {})
            ),
        },
        "raw_results": results,
    }

    if result_json_path:
        Path(result_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

    return normalized