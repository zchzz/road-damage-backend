import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse

from app.config import OUTPUTS_DIR, TASKS_DIR

router = APIRouter(tags=["worker"])


def list_task_files():
    return list(TASKS_DIR.glob("*.json"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


@router.post("/worker/claim")
async def claim_task(worker_id: Optional[str] = Form(None)):
    for task_file in list_task_files():
        task = read_json(task_file)
        if task.get("status") == "pending":
            task["status"] = "processing"
            task["message"] = "任务已被 worker 领取"
            task["worker_id"] = worker_id
            task["updated_at"] = datetime.utcnow().isoformat()
            write_json(task_file, task)

            return {
                "task_id": task["task_id"],
                "filename": task["filename"],
                "saved_filename": task["saved_filename"],
                "confidence": task.get("confidence", 0.25),
                "skip_frames": task.get("skip_frames", 1),
                "mode": task.get("mode", "smoke"),
                "upload_path": task.get("upload_path"),
                "worker_id": worker_id,
            }

    return {
        "task_id": None,
        "message": "暂无待处理任务"
    }


@router.get("/worker/download/{task_id}")
async def download_task_file(task_id: str):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    upload_path = task.get("upload_path")

    if not upload_path:
        raise HTTPException(status_code=400, detail="任务未记录 upload_path")

    file_path = Path(upload_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="原始上传文件不存在")

    return FileResponse(
        path=str(file_path),
        filename=task.get("filename", file_path.name),
        media_type="application/octet-stream"
    )


@router.post("/worker/progress/{task_id}")
async def update_progress(
    task_id: str,
    progress: int = Form(...),
    message: str = Form("处理中"),
    current_frame: int = Form(0),
    total_frames: int = Form(0),
):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    task["status"] = "processing"
    task["progress"] = progress
    task["message"] = message
    task["current_frame"] = current_frame
    task["total_frames"] = total_frames
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    return {"ok": True}


@router.post("/worker/complete/{task_id}")
async def complete_task(
    task_id: str,
    result_video: UploadFile = File(...),
    result_json: Optional[UploadFile] = File(None),
    report_html: Optional[UploadFile] = File(None),
):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)

    output_dir = OUTPUTS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存输出视频
    output_video_name = result_video.filename or "output.mp4"
    output_video_path = output_dir / output_video_name
    with output_video_path.open("wb") as f:
        shutil.copyfileobj(result_video.file, f)

    # 保存 report.html
    report_url = None
    if report_html is not None:
        report_name = report_html.filename or "report.html"
        report_path = output_dir / report_name
        with report_path.open("wb") as f:
            shutil.copyfileobj(report_html.file, f)
        report_url = f"/static/outputs/{task_id}/{report_name}"

    # 保存 result.json，并尽量提取 summary / detections
    summary = {}
    detections = []
    raw_result_payload = {}

    if result_json is not None:
        result_json_name = result_json.filename or "result.json"
        result_json_path = output_dir / result_json_name
        with result_json_path.open("wb") as f:
            shutil.copyfileobj(result_json.file, f)

        try:
            raw_result_payload = json.loads(
                result_json_path.read_text(encoding="utf-8")
            )
            summary = raw_result_payload.get("summary", {})
            detections = raw_result_payload.get("detections", [])
        except Exception:
            raw_result_payload = {}

    output_video_url = f"/static/outputs/{task_id}/{output_video_name}"

    task["status"] = "completed"
    task["progress"] = 100
    task["message"] = "任务处理完成"
    task["result_ready"] = True
    task["output_video_url"] = output_video_url
    task["report_url"] = report_url
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    result_data = {
        "task_id": task_id,
        "status": "completed",
        "summary": summary,
        "detections": detections,
        "output_video_url": output_video_url,
        "report_url": report_url,
        "raw_result": raw_result_payload,
    }
    write_json(output_dir / "result.json", result_data)

    return {
        "ok": True,
        "task_id": task_id,
        "output_video_url": output_video_url,
        "report_url": report_url,
    }


@router.post("/worker/fail/{task_id}")
async def fail_task(
    task_id: str,
    message: str = Form("任务处理失败"),
):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task = read_json(task_file)
    task["status"] = "failed"
    task["message"] = message
    task["updated_at"] = datetime.utcnow().isoformat()
    write_json(task_file, task)

    return {"ok": True}