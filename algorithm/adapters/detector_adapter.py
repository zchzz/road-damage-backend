from pathlib import Path
import json
import os

from algorithm.source_project.road_damage_detector import RoadDamageDetector


BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BASE_DIR / "algorithm" / "weights"
DEFAULT_MODEL_FILE = WEIGHTS_DIR / "best.pt"


def _normalize_class_distribution(class_distribution: dict | None) -> dict:
    if not class_distribution:
        return {}

    return {str(k): int(v) for k, v in class_distribution.items()}


def _resolve_model_path(model_path: str | None = None) -> Path:
    candidates: list[Path] = []

    if model_path:
        candidates.append(Path(model_path).expanduser())

    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        candidates.append(Path(env_model_path).expanduser())

    candidates.append(DEFAULT_MODEL_FILE)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
        if resolved.exists():
            return resolved

    tried_paths = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
        tried_paths.append(str(resolved))

    raise FileNotFoundError(
        "模型权重不存在。已尝试以下路径：\n- " + "\n- ".join(tried_paths)
    )


def detect_video(
    video_path: str,
    output_video_path: str,
    confidence: float = 0.3,
    skip_frames: int = 1,
    model_path: str | None = None,
    report_path: str | None = None,
    result_json_path: str | None = None,
    progress_callback=None,
):
    model_file = _resolve_model_path(model_path)

    output_video = Path(output_video_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    if result_json_path:
        Path(result_json_path).parent.mkdir(parents=True, exist_ok=True)

    detector = RoadDamageDetector(str(model_file))

    results = detector.analyze_video(
        video_path=video_path,
        output_path=str(output_video),
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
        "model_path": str(model_file),
    }

    if result_json_path:
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

    return normalized