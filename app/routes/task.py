import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import TASKS_DIR

router = APIRouter(tags=["task"])

def read_task_file(task_id: str) -> dict:
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    return json.loads(task_file.read_text(encoding="utf-8"))

@router.get("/task/{task_id}")
async def get_task(task_id: str):
    task_data = read_task_file(task_id)

    return {
        "task_id": task_data["task_id"],
        "status": task_data.get("status"),
        "progress": task_data.get("progress", 0),
        "message": task_data.get("message", ""),
        "current_frame": task_data.get("current_frame", 0),
        "total_frames": task_data.get("total_frames", 0),
        "result_ready": task_data.get("result_ready", False),
        "created_at": task_data.get("created_at"),
        "updated_at": task_data.get("updated_at")
    }