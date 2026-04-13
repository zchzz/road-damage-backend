from pathlib import Path
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.task_queue import task_queue

router = APIRouter(prefix="/api/worker", tags=["worker"])


@router.post("/claim")
async def claim_task(worker_id: str = Form(...)):
    task = task_queue.claim_next_task(worker_id)
    if not task:
        return {"task": None}

    return {
        "task": {
            "task_id": task["task_id"],
            "filename": task["filename"],
            "mode": task["mode"],
            "confidence": task["confidence"],
            "skip_frames": task["skip_frames"],
            "download_url": f"/api/worker/download/{task['task_id']}",
        }
    }


@router.get("/download/{task_id}")
async def download_task_file(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    path = Path(task["upload_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="原始视频不存在")

    return FileResponse(path, filename=path.name)


@router.post("/progress/{task_id}")
async def report_progress(
    task_id: str,
    progress: int = Form(...),
    message: str = Form("处理中"),
):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_queue.update_task(task_id, {
        "progress": max(0, min(progress, 100)),
        "message": message,
        "status": "processing",
    })
    return {"ok": True}


@router.post("/complete/{task_id}")
async def complete_task(
    task_id: str,
    result_video: UploadFile = File(...),
    result_json: UploadFile | None = File(None),
    report_html: UploadFile | None = File(None),
):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    output_video_path = Path(task["output_video_path"])
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with output_video_path.open("wb") as f:
        shutil.copyfileobj(result_video.file, f)

    updates = {
        "status": "completed",
        "progress": 100,
        "message": "检测完成",
        "output_video_path": str(output_video_path),
    }

    if result_json is not None:
        json_path = Path(task["result_json_path"])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("wb") as f:
            shutil.copyfileobj(result_json.file, f)
        updates["result_json_path"] = str(json_path)

    if report_html is not None:
        report_path = Path(task["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("wb") as f:
            shutil.copyfileobj(report_html.file, f)
        updates["report_path"] = str(report_path)

    task_queue.update_task(task_id, updates)
    return {"ok": True}


@router.post("/fail/{task_id}")
async def fail_task(
    task_id: str,
    message: str = Form("处理失败"),
):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_queue.update_task(task_id, {
        "status": "failed",
        "message": message,
    })
    return {"ok": True}