from dataclasses import dataclass

import xarray as xr
from river.base import Transformer

from cala.gui.nodes.util import send_through
from cala.streaming.core import Parameters


@dataclass
class FrameCounterParams(Parameters):
    pass

    def validate(self) -> None:
        pass


@dataclass
class FrameCounter(Transformer):
    params: FrameCounterParams
    frame_count_: int = 0

    def learn_one(self, frame: xr.DataArray) -> "FrameCounter":
        self.frame_count_ = frame.coords[self.params.frame_coord].item()

        return self

    def transform_one(self, _: xr.DataArray = None) -> None:
        payload = {
            "type_": "frame_index",
            "index": self.frame_count_,
        }
        send_through(payload)
