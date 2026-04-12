from pydantic import BaseModel
from typing import Optional, Dict, Any


class ResultResponse(BaseModel):
    task_id: str
    status: str
    annotated_video_url: Optional[str] = None
    json_url: Optional[str] = None
    report_url: Optional[str] = None
    summary: Dict[str, Any] = {}
    raw_results: Dict[str, Any] = {}