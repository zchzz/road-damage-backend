import asyncio
from pathlib import Path

from app.config import OUTPUT_DIR, TASK_DATA_DIR
from app.core.detector_runner import run_detection_pipeline
from app.core.stream_manager import stream_manager
from app.core.task_manager import task_manager
from app.core.websocket_manager import ws_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_detect_task(task_id: str) -> None:
    task = task_manager.get_task(task_id)
    if not task:
        logger.error("任务不存在: %s", task_id)
        return

    mode = task.get("mode", "offline")

    try:
        task_manager.update_task(
            task_id,
            {
                "status": "processing",
                "progress": 0,
                "message": "任务开始处理",
            },
        )

        output_video_path = OUTPUT_DIR / task_id / "output.mp4"
        report_path = OUTPUT_DIR / task_id / "report.html"
        result_json_path = TASK_DATA_DIR / task_id / "result.json"
        video_path = Path(task["upload_path"])

        async def progress_callback(payload: dict):
            progress = int(payload.get("progress", 0))
            current_frame = int(payload.get("current_frame", 0))
            total_frames = int(payload.get("total_frames", 0))

            frame_bytes = payload.get("frame_bytes", b"")
            print(f"[stream] task={task_id}, frame_bytes_len={len(frame_bytes) if frame_bytes else 0}")
            if frame_bytes:
                await stream_manager.update_frame(task_id, frame_bytes)

            if total_frames > 0 and current_frame > 0:
                message = f"正在分析第 {current_frame} / {total_frames} 帧"
            elif progress >= 100:
                message = "检测完成，正在整理结果"
            else:
                message = "正在处理视频"

            task_manager.update_task(
                task_id,
                {
                    "status": "processing",
                    "progress": progress,
                    "message": message,
                },
            )

            if mode == "online":
                await ws_manager.send_json(
                    task_id,
                    {
                        "type": payload.get("type", "progress"),
                        "progress": progress,
                        "current_frame": current_frame,
                        "total_frames": total_frames,
                        "detections": payload.get("detections", []),
                        "statistics": payload.get("statistics", {}),
                        "message": message,
                    },
                )

        result = await run_detection_pipeline(
            video_path=video_path,
            output_video_path=output_video_path,
            report_path=report_path,
            result_json_path=result_json_path,
            confidence=task.get("confidence", 0.3),
            skip_frames=task.get("skip_frames", 1),
            progress_callback=progress_callback,
        )

        summary = result.get("summary", {})
        damage_types = summary.get("damage_types", {})

        task_manager.save_result(task_id, result)
        task_manager.update_task(
            task_id,
            {
                "status": "completed",
                "progress": 100,
                "message": "检测完成",
                "output_video_path": str(output_video_path),
                "report_path": str(report_path),
                "result_json_path": str(result_json_path),
                "summary": summary,
                "total_detections": summary.get("total_detections", 0),
                "damage_types": damage_types,
            },
        )

        await stream_manager.mark_finished(task_id)

        if mode == "online":
            await ws_manager.send_json(
                task_id,
                {
                    "type": "completed",
                    "progress": 100,
                    "current_frame": 0,
                    "total_frames": 0,
                    "detections": [],
                    "statistics": damage_types,
                    "message": "检测完成",
                },
            )

        logger.info("任务完成: %s", task_id)

    except Exception as exc:
        logger.exception("任务执行失败: %s", task_id)

        task_manager.update_task(
            task_id,
            {
                "status": "failed",
                "message": str(exc),
            },
        )

        await stream_manager.mark_finished(task_id)

        if mode == "online":
            await ws_manager.send_json(
                task_id,
                {
                    "type": "failed",
                    "message": str(exc),
                },
            )


def start_detect_task(task_id: str) -> None:
    asyncio.create_task(run_detect_task(task_id))