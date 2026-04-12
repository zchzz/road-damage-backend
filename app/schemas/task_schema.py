from pydantic import BaseModel, Field
from typing import Optional, Literal


TaskMode = Literal["online", "offline"]
TaskStatus = Literal["queued", "processing", "completed", "failed"]


class UploadResponse(BaseModel):
    task_id: str
    status: TaskStatus
    mode: TaskMode


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0
    message: str = ""
    mode: Optional[TaskMode] = None