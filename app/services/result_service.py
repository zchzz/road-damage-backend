from app.core.task_manager import task_manager
from app.services.file_service import build_result_urls


def get_result(task_id: str, base_url: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise FileNotFoundError("任务不存在")

    result = task_manager.get_result(task_id)
    urls = build_result_urls(task_id, base_url)

    if not result:
        return {
            "task_id": task_id,
            "status": task.get("status", "queued"),
            "annotated_video_url": None,
            "json_url": None,
            "report_url": None,
            "summary": {},
        }

    summary = result.get("summary", {})
    raw_results = result.get("raw_results", {})

    response = {
        "task_id": task_id,
        "status": task.get("status", "completed"),
        "annotated_video_url": urls["annotated_video_url"],
        "json_url": urls["json_url"],
        "report_url": urls["report_url"],
        "summary": summary,
        "raw_results": raw_results,
    }

    return response