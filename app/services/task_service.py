from app.core.task_queue import task_queue

def get_task_status(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise FileNotFoundError("任务不存在")

    return {
        "task_id": task["task_id"],
        "status": task.get("status", "queued"),
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "mode": task.get("mode"),
    }