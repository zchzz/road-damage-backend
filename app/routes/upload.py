import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import DEFAULT_CONFIDENCE, DEFAULT_SKIP_FRAMES, TASKS_DIR, UPLOADS_DIR

router = APIRouter(tags=["upload"])

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    skip_frames: int = Form(DEFAULT_SKIP_FRAMES),
    mode: str = Form("smoke"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="不支持的视频格式")

    mode = (mode or "smoke").strip().lower()
    if mode not in {"smoke", "real"}:
        mode = "smoke"

    task_id = str(uuid.uuid4())
    saved_filename = f"{task_id}{ext}"
    saved_path = UPLOADS_DIR / saved_filename

    content = await file.read()
    saved_path.write_bytes(content)

    now = datetime.utcnow().isoformat()

    task_data = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建，等待处理",
        "filename": file.filename,
        "saved_filename": saved_filename,
        "upload_path": str(saved_path),
        "mode": mode,
        "confidence": confidence,
        "skip_frames": skip_frames,
        "current_frame": 0,
        "total_frames": 0,
        "result_ready": False,
        "created_at": now,
        "updated_at": now
    }

    task_file = TASKS_DIR / f"{task_id}.json"
    task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务创建成功",
        "mode": mode,
    }