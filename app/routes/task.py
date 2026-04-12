from fastapi import APIRouter, HTTPException

from app.schemas.task_schema import TaskStatusResponse
from app.services.task_service import get_task_status

router = APIRouter(prefix="/api", tags=["task"])


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def query_task_status(task_id: str):
    try:
        data = get_task_status(task_id)
        return TaskStatusResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc