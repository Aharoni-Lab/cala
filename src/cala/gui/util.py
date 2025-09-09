from queue import Queue


class QManager:
    _qs: dict[str, Queue] = {}

    @classmethod
    def get_queue(cls, name: str) -> Queue:
        if cls._qs.get(name) is None:
            cls._qs[name] = Queue()
        return cls._qs.get(name)


# def send_through(payload: dict) -> None:
#     manager = get_socket_manager()
#
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError as e:
#         print(e)
#         loop = None
#
#     try:
#         for websocket in manager.active_connections:
#             message = WebsocketMessage(payload=payload)
#
#             if loop:
#                 loop.create_task(manager.send_json(message.model_dump(mode="json"), websocket))
#             else:
#                 asyncio.run(manager.send_json(message.model_dump(mode="json"), websocket))
#
#     # https://fastapi.tiangolo.com/reference/websockets/#fastapi.WebSocket.send
#     except WebSocketDisconnect:
#         print(f"Num Connections: {len(manager.active_connections)}")
#         if websocket in manager.active_connections:
#             manager.disconnect(websocket)
#         pass
