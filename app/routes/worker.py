import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.config import OUTPUTS_DIR, TASKS_DIR

router = APIRouter(tags=["worker"])

def list_task_files():
    return list(TASKS_DIR.glob("*.json"))

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@router.post("/worker/claim")
async def claim_task():
    for task_file in list_task_files():
        task = read_json(task_file)
        if task.get("status") == "pending":
            task["status"] = "processing"
            task["message"] = "任务已被 worker 领取"
            task["updated_at"] = datetime.utcnow().isoformat()
            write_json(task_file, task)

            return {
                "task_id": task["task_id"],
                "filename": task["filename"],
                "saved_filename": task["saved_filename"],
                "confidence": task.get("confidence", 0.25),
                "skip_frames": task.get("skip_frames", 1),
                "upload_path": task.get("upload_path")
            }

    return {
        "task_id": None,
        "message": "暂无待处理任务"
    }

@router.post("/worker/progress/{task_id}")
async def update_progress(task_id: str, payload: dict):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    task["status"] = "processing"
    task["progress"] = payload.get("progress", task.get("progress", 0))
    task["message"] = payload.get("message", task.get("message", "处理中"))
    task["current_frame"] = payload.get("current_frame", task.get("current_frame", 0))
    task["total_frames"] = payload.get("total_frames", task.get("total_frames", 0))
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    return {"ok": True}

@router.post("/worker/complete/{task_id}")
async def complete_task(task_id: str, payload: dict):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    task["status"] = "completed"
    task["progress"] = 100
    task["message"] = payload.get("message", "任务处理完成")
    task["result_ready"] = True
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    output_dir = OUTPUTS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "task_id": task_id,
        "status": "completed",
        "summary": payload.get("summary", {}),
        "detections": payload.get("detections", []),
        "output_video_url": payload.get("output_video_url"),
        "report_url": payload.get("report_url"),
    }
    write_json(output_dir / "result.json", result_data)

    return {"ok": True}

@router.post("/worker/fail/{task_id}")
async def fail_task(task_id: str, payload: dict):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    task["status"] = "failed"
    task["message"] = payload.get("message", "任务处理失败")
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    return {"ok": True}