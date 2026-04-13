import asyncio
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.adapters.detector_adapter import detect_video


async def run_detection_pipeline(
    video_path: Path,
    output_video_path: Path,
    report_path: Path,
    result_json_path: Path,
    confidence: float,
    skip_frames: int,
    progress_callback=None,
):
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def sync_progress_callback(payload: dict):
        if progress_callback:
            asyncio.run_coroutine_threadsafe(progress_callback(payload), loop)

    result = await asyncio.to_thread(
        detect_video,
        video_path=str(video_path),
        output_video_path=str(output_video_path),
        confidence=confidence,
        skip_frames=skip_frames,
        report_path=str(report_path),
        result_json_path=str(result_json_path),
        progress_callback=sync_progress_callback,
    )
    return result