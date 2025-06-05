import asyncio
from dataclasses import dataclass

import xarray as xr
from fastapi import WebSocketDisconnect
from river.base import Transformer

from cala.gui import WebsocketMessage
from cala.gui.dependencies import get_socket_manager
from cala.streaming.core import Axis, Parameters


@dataclass
class FrameCounterParams(Axis, Parameters):
    pass

    def validate(self) -> None:
        pass


@dataclass
class FrameCounter(Transformer):
    params: FrameCounterParams
    frame_count_: int = 0

    def learn_one(self, frame: xr.DataArray) -> "FrameCounter":
        self.frame_count_ = frame.coords[Axis.frame_coordinates].item()

        return self

    def transform_one(self, _: xr.DataArray = None) -> None:
        payload = {
            "type_": "frame_index",
            "index": self.frame_count_,
        }
        send_through(payload)


def send_through(payload: dict) -> None:
    manager = get_socket_manager()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        print(e)
        loop = None

    try:
        for websocket in manager.active_connections:
            message = WebsocketMessage(payload=payload)

            if loop:
                loop.create_task(manager.send_json(message.model_dump(mode="json"), websocket))
            else:
                asyncio.run(manager.send_json(message.model_dump(mode="json"), websocket))

    # https://fastapi.tiangolo.com/reference/websockets/#fastapi.WebSocket.send
    except WebSocketDisconnect:
        print(f"Num Connections: {len(manager.active_connections)}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)
        pass
