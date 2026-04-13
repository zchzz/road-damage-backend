import json

from fastapi import APIRouter, HTTPException

from app.config import OUTPUTS_DIR, TASKS_DIR

router = APIRouter(tags=["result"])

@router.get("/result/{task_id}")
async def get_result(task_id: str):
    task_file = TASKS_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_file.read_text(encoding="utf-8"))

    if task_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    result_file = OUTPUTS_DIR / task_id / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    result_data = json.loads(result_file.read_text(encoding="utf-8"))
    return result_data