from pathlib import Path
from typing import Optional, Dict, Any

from app.config import TASK_DATA_DIR
from app.utils.json_utils import read_json, write_json
from app.utils.time_utils import now_str


class TaskManager:
    def __init__(self, task_data_dir: Path):
        self.task_data_dir = task_data_dir

    def get_task_dir(self, task_id: str) -> Path:
        return self.task_data_dir / task_id

    def get_meta_path(self, task_id: str) -> Path:
        return self.get_task_dir(task_id) / "meta.json"

    def get_result_path(self, task_id: str) -> Path:
        return self.get_task_dir(task_id) / "result.json"

    def create_task(self, meta: Dict[str, Any]) -> None:
        meta["created_at"] = now_str()
        meta["updated_at"] = now_str()
        write_json(self.get_meta_path(meta["task_id"]), meta)

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        meta_path = self.get_meta_path(task_id)
        meta = read_json(meta_path, default={})
        if not meta:
            raise FileNotFoundError("任务不存在")

        meta.update(updates)
        meta["updated_at"] = now_str()
        write_json(meta_path, meta)
        return meta

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return read_json(self.get_meta_path(task_id), default=None)

    def save_result(self, task_id: str, result: Dict[str, Any]) -> None:
        write_json(self.get_result_path(task_id), result)

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        return read_json(self.get_result_path(task_id), default=None)


task_manager = TaskManager(TASK_DATA_DIR)