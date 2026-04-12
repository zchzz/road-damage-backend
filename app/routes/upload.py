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

    upload_path = task_upload_dir / file.filename

    try:
        print(f"[upload] 5. start saving file -> {upload_path}")
        await save_upload_file(file, upload_path)
        print("[upload] 6. file saved")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc

    meta = {
        "task_id": task_id,
        "filename": file.filename,
        "mode": mode,
        "status": "queued",
        "progress": 0,
        "message": "任务已创建",
        "confidence": confidence,
        "skip_frames": skip_frames,
        "upload_path": str(upload_path),
    }

    print("[upload] 7. creating task meta")
    task_manager.create_task(meta)
    print("[upload] 8. task meta created")

    background_tasks.add_task(run_detect_task, task_id)
    print("[upload] 9. background task added")

    print("[upload] 10. returning response")
    return UploadResponse(
        task_id=task_id,
        status="queued",
        mode=mode,
    )