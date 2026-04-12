from fastapi import APIRouter, HTTPException, Request

from app.schemas.result_schema import ResultResponse
from app.services.result_service import get_result

router = APIRouter(prefix="/api", tags=["result"])


@router.get("/result/{task_id}", response_model=ResultResponse)
async def query_task_result(task_id: str, request: Request):
    try:
        base_url = str(request.base_url).rstrip("/")
        data = get_result(task_id, base_url)
        return ResultResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc