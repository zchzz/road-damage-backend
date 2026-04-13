from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.task_queue import task_queue

router = APIRouter(prefix="/api", tags=["result-file"])


@router.get("/result-file/{task_id}/video")
async def get_result_video(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    path = Path(task.get("output_video_path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果视频不存在")

    return FileResponse(path, filename=path.name)


@router.get("/result-file/{task_id}/json")
async def get_result_json(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    path = Path(task.get("result_json_path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果 JSON 不存在")

    return FileResponse(path, filename=path.name, media_type="application/json")


@router.get("/result-file/{task_id}/report")
async def get_result_report(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    path = Path(task.get("report_path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")

    return FileResponse(path, filename=path.name, media_type="text/html")