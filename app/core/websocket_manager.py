from collections import defaultdict
from typing import Dict, Set
from fastapi import WebSocket


class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = defaultdict(set)

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[task_id].add(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        if task_id in self.connections:
            self.connections[task_id].discard(websocket)
            if not self.connections[task_id]:
                del self.connections[task_id]

    async def send_json(self, task_id: str, payload: dict) -> None:
        if task_id not in self.connections:
            return

        dead_connections = []
        for websocket in self.connections[task_id]:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead_connections.append(websocket)

        for websocket in dead_connections:
            self.disconnect(task_id, websocket)


ws_manager = WebSocketManager()