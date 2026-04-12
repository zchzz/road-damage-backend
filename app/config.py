from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/outputs"))
TASK_DATA_DIR = Path(os.getenv("TASK_DATA_DIR", "/tmp/task_data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/tmp/logs"))

APP_NAME = "Road Damage Detection Backend"
APP_VERSION = "1.0.0"

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.3"))
DEFAULT_SKIP_FRAMES = int(os.getenv("DEFAULT_SKIP_FRAMES", "1"))

def ensure_directories() -> None:
    for directory in [UPLOAD_DIR, OUTPUT_DIR, TASK_DATA_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)