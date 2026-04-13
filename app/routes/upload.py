import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SKIP_FRAMES,
    MAX_FILE_SIZE_MB,
    TASKS_DIR,
    UPLOADS_DIR,
)

router = APIRouter(tags=["upload"])

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CHUNK_SIZE = 1024 * 1024  # 1MB


async def save_upload_file_chunked(
    file: UploadFile,
    destination: Path,
    max_size_bytes: int,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    with destination.open("wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break

            total_written += len(chunk)
            if total_written > max_size_bytes:
                f.close()
                try:
                    destination.unlink(missing_ok=True)
                except Exception:
                    pass

                raise HTTPException(
                    status_code=413,
                    detail=f"上传文件过大，当前限制为 {MAX_FILE_SIZE_MB}MB",
                )

            f.write(chunk)

    await file.close()
    return total_written


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
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="不支持的视频格式")

    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=400, detail="confidence 必须在 0 到 1 之间")

    if int(skip_frames) < 1:
        raise HTTPException(status_code=400, detail="skip_frames 必须大于等于 1")

    mode = (mode or "smoke").strip().lower()
    if mode not in {"smoke", "real"}:
        mode = "smoke"

    task_id = str(uuid.uuid4())
    saved_filename = f"{task_id}{ext}"
    saved_path = UPLOADS_DIR / saved_filename

    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    file_size = await save_upload_file_chunked(
        file=file,
        destination=saved_path,
        max_size_bytes=max_size_bytes,
    )

    now = datetime.now(timezone.utc).isoformat()
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建，等待处理",
        "filename": file.filename,
        "saved_filename": saved_filename,
        "upload_path": str(saved_path),
        "file_size": file_size,
        "mode": mode,
        "confidence": float(confidence),
        "skip_frames": int(skip_frames),
        "current_frame": 0,
        "total_frames": 0,
        "result_ready": False,
        "created_at": now,
        "updated_at": now,
    }

    task_file = TASKS_DIR / f"{task_id}.json"
    task_file.write_text(
        json.dumps(task_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务创建成功",
        "mode": mode,
    }