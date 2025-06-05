from fastapi import WebSocket

from cala.gui import WebsocketMessage


class SocketManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def send_json(self, data: dict, websocket: WebSocket) -> None:
        data = WebsocketMessage.model_validate(data)
        await websocket.send_json(data)

    async def broadcast(self, data: dict) -> None:
        for connection in self.active_connections:
            await connection.send_json(data)
