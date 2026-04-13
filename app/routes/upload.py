from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SKIP_FRAMES,
    OUTPUT_DIR,
    TASK_DATA_DIR,
    UPLOAD_DIR,
)
from app.core.task_manager import task_manager
from app.schemas.task_schema import UploadResponse
from app.services.detect_service import run_detect_task
from app.utils.file_utils import (
    create_task_directories,
    generate_task_id,
    save_upload_file,
    validate_video_file,
)

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    skip_frames: int = Form(DEFAULT_SKIP_FRAMES),
):
    print("[upload] 1. request arrived")

    if mode not in {"online", "offline"}:
        raise HTTPException(status_code=400, detail="mode 必须是 online 或 offline")

    try:
        print(f"[upload] 2. validating file: {file.filename}")
        validate_video_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = generate_task_id()
    print(f"[upload] 3. generated task_id: {task_id}")

    task_upload_dir, task_output_dir, task_meta_dir = create_task_directories(
        task_id, UPLOAD_DIR, OUTPUT_DIR, TASK_DATA_DIR
    )
    print("[upload] 4. directories created")
from fastapi import APIRouter, File, Form, UploadFile
from app.config import DEFAULT_CONFIDENCE, DEFAULT_SKIP_FRAMES, OUTPUT_DIR, TASK_DATA_DIR, UPLOAD_DIR
from app.schemas.task_schema import UploadResponse
from app.core.task_queue import task_queue
from app.utils.file_utils import (
    create_task_directories,
    generate_task_id,
    save_upload_file,
    validate_video_file,
)

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    mode: str = Form(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    skip_frames: int = Form(DEFAULT_SKIP_FRAMES),
):
    validate_video_file(file)

    task_id = generate_task_id()
    task_upload_dir, task_output_dir, task_meta_dir = create_task_directories(
        task_id, UPLOAD_DIR, OUTPUT_DIR, TASK_DATA_DIR
    )

    upload_path = task_upload_dir / file.filename
    await save_upload_file(file, upload_path)

    meta = {
        "task_id": task_id,
        "filename": file.filename,
        "mode": mode,
        "status": "queued",
        "progress": 0,
        "message": "已上传，等待 worker 处理",
        "upload_path": str(upload_path),
        "output_video_path": str(task_output_dir / "output.mp4"),
        "report_path": str(task_output_dir / "report.html"),
        "result_json_path": str(task_meta_dir / "result.json"),
        "confidence": confidence,
        "skip_frames": skip_frames,
    }
    task_queue.create_task(meta)

    return {
        "task_id": task_id,
        "status": "queued",
        "mode": mode,
    }