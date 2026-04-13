from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import TASKS_DIR

router = APIRouter(tags=["ws"])


POLL_INTERVAL_SECONDS = 1.5
FINAL_HOLD_SECONDS = 3.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_task_file(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def read_task(task_id: str) -> dict | None:
    task_file = get_task_file(task_id)
    if not task_file.exists():
        return None
    return read_json(task_file)


def normalize_task_payload(task_id: str, task: dict | None) -> dict:
    if not task:
        return {
            "type": "task_update",
            "task_id": task_id,
            "status": "not_found",
            "progress": 0,
            "message": "任务不存在",
            "current_frame": 0,
            "total_frames": 0,
            "result_ready": False,
            "filename": None,
            "worker_id": None,
            "mode": None,
            "confidence": None,
            "skip_frames": None,
            "output_video_url": None,
            "result_json_url": None,
            "report_url": None,
            "annotated_video_url": None,
            "json_url": None,
            "html_report_url": None,
            "created_at": None,
            "updated_at": None,
            "claimed_at": None,
        }

    output_video_url = task.get("output_video_url")
    result_json_url = task.get("result_json_url")
    report_url = task.get("report_url")

    return {
        "type": "task_update",
        "task_id": task.get("task_id", task_id),
        "status": task.get("status", "pending"),
        "progress": int(task.get("progress", 0) or 0),
        "message": task.get("message", ""),
        "current_frame": int(task.get("current_frame", 0) or 0),
        "total_frames": int(task.get("total_frames", 0) or 0),
        "result_ready": bool(task.get("result_ready", False)),
        "filename": task.get("filename"),
        "worker_id": task.get("worker_id"),
        "mode": task.get("mode", "real"),
        "confidence": task.get("confidence", 0.25),
        "skip_frames": task.get("skip_frames", 1),
        "output_video_url": output_video_url,
        "result_json_url": result_json_url,
        "report_url": report_url,
        # 前端兼容别名
        "annotated_video_url": output_video_url,
        "json_url": result_json_url,
        "html_report_url": report_url,
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "claimed_at": task.get("claimed_at"),
    }


@router.websocket("/ws/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()

    last_payload = None
    final_payload_sent = False

    try:
        while True:
            task = read_task(task_id)
            payload = normalize_task_payload(task_id, task)

            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload

            status = payload.get("status")

            if status == "not_found":
                break

            if status in {"completed", "failed"}:
                if not final_payload_sent:
                    final_payload_sent = True
                    await asyncio.sleep(FINAL_HOLD_SECONDS)
                break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.send_json({
                "type": "task_update",
                "task_id": task_id,
                "status": "error",
                "progress": 0,
                "message": "WebSocket 推送异常中断",
                "current_frame": 0,
                "total_frames": 0,
                "result_ready": False,
                "filename": None,
                "worker_id": None,
                "mode": None,
                "confidence": None,
                "skip_frames": None,
                "output_video_url": None,
                "result_json_url": None,
                "report_url": None,
                "annotated_video_url": None,
                "json_url": None,
                "html_report_url": None,
                "created_at": None,
                "updated_at": None,
                "claimed_at": None,
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass