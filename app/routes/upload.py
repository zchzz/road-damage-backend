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


# 修改后的 upload_video 函数关键部分
@router.post("/upload", response_model=UploadResponse)
async def upload_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: str = Form(...),
        confidence: float = Form(DEFAULT_CONFIDENCE),
        skip_frames: int = Form(DEFAULT_SKIP_FRAMES),
):
    # 1. 生成唯一 ID
    task_id = generate_task_id()

    # 2. 创建持久化目录 (Render 部署建议检查目录权限)
    task_upload_dir, task_output_dir, task_meta_dir = create_task_directories(
        task_id, UPLOAD_DIR, OUTPUT_DIR, TASK_DATA_DIR
    )

    # 3. 保存原始视频到存储目录
    upload_path = task_upload_dir / file.filename
    await save_upload_file(file, upload_path)

    # 4. 构建任务元数据 (这就是你的“轻量持久化”存储)
    meta = {
        "task_id": task_id,
        "filename": file.filename,
        "mode": mode,
        "status": "processing",  # 初始状态
        "progress": 0,
        "upload_path": f"/static/tasks/{task_id}/uploads/{file.filename}",  # 对应 main.py 的静态映射
        "result_path": f"/static/outputs/{task_id}/result.mp4",  # 预留结果路径
        "meta_path": f"/static/tasks/{task_id}/meta.json"
    }

    # 保存 meta 到磁盘
    task_manager.save_task_meta(task_id, meta)

    # 5. 启动后台推理任务
    background_tasks.add_task(
        run_detect_task,
        task_id=task_id,
        video_path=str(upload_path),
        output_dir=str(task_output_dir),
        confidence=confidence
    )

    return {
        "task_id": task_id,
        "message": "视频上传成功，后台处理中...",
        "status": "success"
    }