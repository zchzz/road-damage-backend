from fastapi import APIRouter, File, Form, UploadFile

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SKIP_FRAMES,
    OUTPUT_DIR,
    TASK_DATA_DIR,
    UPLOAD_DIR,
)
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
    """
    前端上传视频：
    1. 保存原视频
    2. 写入 SQLite 任务队列
    3. 返回 task_id
    注意：这里不再进行服务端检测
    """
    validate_video_file(file)

    task_id = generate_task_id()

    task_upload_dir, task_output_dir, task_meta_dir = create_task_directories(
        task_id=task_id,
        upload_root=UPLOAD_DIR,
        output_root=OUTPUT_DIR,
        task_root=TASK_DATA_DIR,
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
        "worker_id": "",
    }

    task_queue.create_task(meta)

    return UploadResponse(
        task_id=task_id,
        status="queued",
        mode=mode,
    )