import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import TASKS_DIR

router = APIRouter(tags=["ws"])

def read_task(task_id: str):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        return None
    return json.loads(task_file.read_text(encoding="utf-8"))

@router.websocket("/ws/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()

    try:
        last_payload = None

        while True:
            task = read_task(task_id)
            if not task:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": "not_found",
                    "message": "任务不存在"
                })
                break

            payload = {
                "task_id": task["task_id"],
                "status": task.get("status"),
                "progress": task.get("progress", 0),
                "message": task.get("message", ""),
                "current_frame": task.get("current_frame", 0),
                "total_frames": task.get("total_frames", 0),
                "result_ready": task.get("result_ready", False),
                "updated_at": task.get("updated_at"),
            }

            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload

            if task.get("status") in {"completed", "failed"}:
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        return