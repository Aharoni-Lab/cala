import asyncio
import random
from dataclasses import dataclass

import xarray as xr
from fastapi import WebSocketDisconnect
from river.base import Transformer

from cala.gui.dependencies import get_socket_manager
from cala.streaming.core import Parameters


@dataclass
class MetricStreamerParams(Parameters):
    idx: int = 0

    def validate(self) -> None:
        pass


@dataclass
class MetricStreamer(Transformer):
    params: MetricStreamerParams

    def learn_one(self, frame: xr.DataArray) -> None: ...

    def transform_one(self, frame: xr.DataArray) -> None:
        manager = get_socket_manager()

        try:
            loop = asyncio.get_running_loop()

            for websocket in manager.active_connections:
                payload = {"index": self.params.idx, "value": int(random.random() * 10000) / 100}
                loop.create_task(manager.send_json(payload, websocket))

                # await manager.send_json(
                #     {"index": self.params.idx, "value": int(random.random() * 10000) / 100},
                #     websocket,
                # )
        # https://fastapi.tiangolo.com/reference/websockets/#fastapi.WebSocket.send
        except WebSocketDisconnect:
            print(f"Num Connections: {len(manager.active_connections)}")
            if websocket in manager.active_connections:
                manager.disconnect(websocket)
            pass

        self.params.idx += 1
