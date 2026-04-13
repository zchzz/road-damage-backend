from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import TASKS_DIR, UPLOADS_DIR

router = APIRouter(tags=["upload"])


ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


def utcnow() -> str:
    return datetime.utcnow().isoformat()


def write_json(path: Path, data: dict):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "video.mp4").name.strip()
    return name or "video.mp4"


def validate_video_file(file: UploadFile):
    filename = sanitize_filename(file.filename or "video.mp4")
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{suffix or 'unknown'}，请上传视频文件",
        )


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    skip_frames: int = Form(1),
    mode: Optional[str] = Form("real"),
):
    validate_video_file(file)

    filename = sanitize_filename(file.filename or "video.mp4")
    task_id = uuid.uuid4().hex

    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="confidence 必须在 0 到 1 之间")

    if skip_frames < 1:
        raise HTTPException(status_code=400, detail="skip_frames 必须大于等于 1")

    normalized_mode = str(mode or "real").strip().lower()
    if normalized_mode not in {"real", "smoke"}:
        normalized_mode = "real"

    task_upload_dir = UPLOADS_DIR / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_filename = f"input{Path(filename).suffix.lower() or '.mp4'}"
    saved_path = task_upload_dir / saved_filename

    try:
        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {e!r}")

    task_data = {
        "task_id": task_id,
        "status": "pending",
        "message": "文件上传成功，等待 worker 处理",
        "progress": 0,
        "result_ready": False,
        "filename": filename,
        "saved_filename": saved_filename,
        "upload_path": str(saved_path.resolve()),
        "upload_url": f"/static/uploads/{task_id}/{saved_filename}",
        "confidence": float(confidence),
        "skip_frames": int(skip_frames),
        "mode": normalized_mode,
        "current_frame": 0,
        "total_frames": 0,
        "worker_id": None,
        "output_video_url": None,
        "result_json_url": None,
        "report_url": None,
        "created_at": utcnow(),
        "updated_at": utcnow(),
    }

    task_file = TASKS_DIR / f"{task_id}.json"
    write_json(task_file, task_data)

    return {
        "task_id": task_id,
        "status": task_data["status"],
        "message": task_data["message"],
        "progress": task_data["progress"],
        "filename": task_data["filename"],
        "upload_url": task_data["upload_url"],
        "confidence": task_data["confidence"],
        "skip_frames": task_data["skip_frames"],
        "mode": task_data["mode"],
        "created_at": task_data["created_at"],
    }