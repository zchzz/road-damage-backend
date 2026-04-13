from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"

for path in [DATA_DIR, TASKS_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def _parse_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


CORS_ORIGINS = _parse_csv_env(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1024"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_SKIP_FRAMES = int(os.getenv("DEFAULT_SKIP_FRAMES", "1"))
DEFAULT_TASK_MODE = os.getenv("DEFAULT_TASK_MODE", "real").strip().lower()

if DEFAULT_TASK_MODE not in {"real", "smoke"}:
    DEFAULT_TASK_MODE = "real"

# 公网 worker 鉴权
WORKER_SHARED_SECRET = os.getenv("WORKER_SHARED_SECRET", "").strip()

# 模型路径：供 detector_adapter / worker 环境统一读取
MODEL_PATH = os.getenv("MODEL_PATH", "algorithm/weights/best.pt").strip()

# 一些可选的运行信息
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", "").rstrip("/")

# 上传大小上限（字节）
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024