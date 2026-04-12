import asyncio


class StreamManager:
    def __init__(self):
        self._latest_frames: dict[str, bytes] = {}
        self._finished: set[str] = set()
        self._lock = asyncio.Lock()

    async def update_frame(self, task_id: str, frame_bytes: bytes):
        async with self._lock:
            self._latest_frames[task_id] = frame_bytes

    async def get_frame(self, task_id: str) -> bytes | None:
        async with self._lock:
            return self._latest_frames.get(task_id)

    async def mark_finished(self, task_id: str):
        async with self._lock:
            self._finished.add(task_id)

    async def is_finished(self, task_id: str) -> bool:
        async with self._lock:
            return task_id in self._finished

    async def clear(self, task_id: str):
        async with self._lock:
            self._latest_frames.pop(task_id, None)
            self._finished.discard(task_id)


stream_manager = StreamManager()