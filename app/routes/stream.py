import asyncio
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.core.stream_manager import stream_manager

router = APIRouter(prefix="/api", tags=["stream"])


async def mjpeg_generator(task_id: str):
    print(f"[mjpeg] generator started, task_id={task_id}")
    finished_at = None

    while True:
        frame = await stream_manager.get_frame(task_id)
        finished = await stream_manager.is_finished(task_id)

        print(
            f"[mjpeg] task_id={task_id}, has_frame={frame is not None}, "
            f"frame_len={len(frame) if frame else 0}, finished={finished}"
        )

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

        if finished and finished_at is None:
            finished_at = time.time()
            print(f"[mjpeg] task finished, enter grace period, task_id={task_id}")

        if finished_at is not None and time.time() - finished_at > 5:
            print(f"[mjpeg] grace period ended, stop generator, task_id={task_id}")
            break

        await asyncio.sleep(0.1)

    print(f"[mjpeg] generator ended, task_id={task_id}")


@router.get("/stream/{task_id}")
async def stream_video(task_id: str):
    print(f"[mjpeg] /api/stream called, task_id={task_id}")
    return StreamingResponse(
        mjpeg_generator(task_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )