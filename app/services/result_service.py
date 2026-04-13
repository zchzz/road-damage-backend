from pathlib import Path

from app.core.task_queue import task_queue


def get_result(task_id: str, base_url: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise FileNotFoundError("任务不存在")

    status = task.get("status", "queued")
    output_video_path = task.get("output_video_path", "")
    report_path = task.get("report_path", "")
    result_json_path = task.get("result_json_path", "")

    annotated_video_url = None
    report_url = None
    json_url = None

    if output_video_path and Path(output_video_path).exists():
        annotated_video_url = f"{base_url}/api/result-file/{task_id}/video"

    if report_path and Path(report_path).exists():
        report_url = f"{base_url}/api/result-file/{task_id}/report"

    if result_json_path and Path(result_json_path).exists():
        json_url = f"{base_url}/api/result-file/{task_id}/json"

    return {
        "task_id": task_id,
        "status": status,
        "annotated_video_url": annotated_video_url,
        "json_url": json_url,
        "report_url": report_url,
        "summary": {},
        "raw_results": {},
    }