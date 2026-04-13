import sqlite3
import json
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

from app.config import TASK_DATA_DIR
from app.utils.time_utils import now_str

DB_PATH = TASK_DATA_DIR / "tasks.db"


class SQLiteTaskQueue:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT DEFAULT '',
                upload_path TEXT NOT NULL,
                output_video_path TEXT DEFAULT '',
                report_path TEXT DEFAULT '',
                result_json_path TEXT DEFAULT '',
                confidence REAL DEFAULT 0.3,
                skip_frames INTEGER DEFAULT 1,
                worker_id TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            conn.commit()

    def create_task(self, meta: Dict[str, Any]) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("""
            INSERT INTO tasks (
                task_id, filename, mode, status, progress, message,
                upload_path, output_video_path, report_path, result_json_path,
                confidence, skip_frames, worker_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meta["task_id"],
                meta["filename"],
                meta.get("mode", "offline"),
                meta.get("status", "queued"),
                meta.get("progress", 0),
                meta.get("message", ""),
                meta["upload_path"],
                meta.get("output_video_path", ""),
                meta.get("report_path", ""),
                meta.get("result_json_path", ""),
                meta.get("confidence", 0.3),
                meta.get("skip_frames", 1),
                meta.get("worker_id", ""),
                now_str(),
                now_str(),
            ))
            conn.commit()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise FileNotFoundError("任务不存在")

        task.update(updates)
        task["updated_at"] = now_str()

        with self._lock, self._get_conn() as conn:
            conn.execute("""
            UPDATE tasks SET
                filename = ?, mode = ?, status = ?, progress = ?, message = ?,
                upload_path = ?, output_video_path = ?, report_path = ?, result_json_path = ?,
                confidence = ?, skip_frames = ?, worker_id = ?, updated_at = ?
            WHERE task_id = ?
            """, (
                task["filename"],
                task["mode"],
                task["status"],
                task["progress"],
                task["message"],
                task["upload_path"],
                task.get("output_video_path", ""),
                task.get("report_path", ""),
                task.get("result_json_path", ""),
                task.get("confidence", 0.3),
                task.get("skip_frames", 1),
                task.get("worker_id", ""),
                task["updated_at"],
                task_id,
            ))
            conn.commit()

        return self.get_task(task_id)

    def claim_next_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM tasks
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
            """).fetchone()

            if not row:
                return None

            task_id = row["task_id"]
            conn.execute("""
                UPDATE tasks
                SET status = 'processing',
                    message = 'worker 已接单',
                    worker_id = ?,
                    updated_at = ?
                WHERE task_id = ? AND status = 'queued'
            """, (worker_id, now_str(), task_id))
            conn.commit()

        return self.get_task(task_id)


task_queue = SQLiteTaskQueue(DB_PATH)