from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile

from app.config import ALLOWED_VIDEO_SUFFIXES, MAX_FILE_SIZE_MB
from app.utils.time_utils import now_str


def generate_task_id() -> str:
    return f"task_{uuid4().hex[:12]}"


def validate_video_file(file: UploadFile) -> None:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"不支持的文件格式: {suffix}")


async def save_upload_file(file: UploadFile, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total_size = 0
    max_size = MAX_FILE_SIZE_MB * 1024 * 1024
    chunk_count = 0

    with save_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                print(f"[save_upload_file] done, total_size={total_size}, chunks={chunk_count}")
                break

            chunk_count += 1
            total_size += len(chunk)

            if chunk_count % 10 == 0:
                print(f"[save_upload_file] chunks={chunk_count}, total_size={total_size}")

            if total_size > max_size:
                raise ValueError(f"文件大小超过限制，最大 {MAX_FILE_SIZE_MB}MB")

            f.write(chunk)


def create_task_directories(task_id: str, upload_dir: Path, output_dir: Path, task_data_dir: Path):
    task_upload_dir = upload_dir / task_id
    task_output_dir = output_dir / task_id
    task_meta_dir = task_data_dir / task_id

    task_upload_dir.mkdir(parents=True, exist_ok=True)
    task_output_dir.mkdir(parents=True, exist_ok=True)
    task_meta_dir.mkdir(parents=True, exist_ok=True)

    return task_upload_dir, task_output_dir, task_meta_dir