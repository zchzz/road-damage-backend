from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import OUTPUTS_DIR, TASKS_DIR

router = APIRouter(tags=["result"])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_task_file(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def get_output_dir(task_id: str) -> Path:
    return OUTPUTS_DIR / task_id


def normalize_result_payload(task_id: str, task_data: dict, stored_result: dict | None = None) -> dict:
    stored_result = stored_result or {}

    output_video_url = (
        stored_result.get("output_video_url")
        or task_data.get("output_video_url")
    )
    result_json_url = (
        stored_result.get("result_json_url")
        or task_data.get("result_json_url")
    )
    report_url = (
        stored_result.get("report_url")
        or task_data.get("report_url")
    )

    summary = stored_result.get("summary", {})
    detections = stored_result.get("detections", [])
    raw_result = stored_result.get("raw_result", {})

    return {
        "task_id": task_id,
        "status": task_data.get("status", "pending"),
        "message": task_data.get("message", ""),
        "progress": task_data.get("progress", 0),
        "result_ready": bool(task_data.get("result_ready", False)),
        "summary": summary,
        "detections": detections,
        "output_video_url": output_video_url,
        "result_json_url": result_json_url,
        "report_url": report_url,
        "annotated_video_url": output_video_url,
        "json_url": result_json_url,
        "html_report_url": report_url,
        "raw_result": raw_result,
        "current_frame": task_data.get("current_frame", 0),
        "total_frames": task_data.get("total_frames", 0),
        "updated_at": task_data.get("updated_at"),
        "created_at": task_data.get("created_at"),
    }


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    task_file = get_task_file(task_id)
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = read_json(task_file)

    output_dir = get_output_dir(task_id)
    result_file = output_dir / "result.json"

    if task_data.get("status") != "completed":
        return normalize_result_payload(task_id, task_data, stored_result=None)

    if not result_file.exists():
        return normalize_result_payload(task_id, task_data, stored_result=None)

    try:
        stored_result = read_json(result_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取结果文件失败: {e!r}")

    return normalize_result_payload(task_id, task_data, stored_result=stored_result)