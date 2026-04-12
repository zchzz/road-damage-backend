def build_result_urls(task_id: str, base_url: str) -> dict:
    return {
        "annotated_video_url": f"{base_url}/static/outputs/{task_id}/output.mp4",
        "json_url": f"{base_url}/static/tasks/{task_id}/result.json",
        "report_url": f"{base_url}/static/outputs/{task_id}/report.html",
    }