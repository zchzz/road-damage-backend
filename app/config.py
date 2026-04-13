import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"

for path in [DATA_DIR, TASKS_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1024"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_SKIP_FRAMES = int(os.getenv("DEFAULT_SKIP_FRAMES", "1"))