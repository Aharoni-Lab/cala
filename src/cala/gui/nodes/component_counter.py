from dataclasses import dataclass

import xarray as xr
from river.base import Transformer

from cala.gui.nodes.util import send_through
from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints


@dataclass
class ComponentCounterParams(Axis, Parameters):
    pass

    def validate(self) -> None:
        pass


@dataclass
class ComponentCounter(Transformer):
    params: ComponentCounterParams
    component_count_: int = 0

    def learn_one(self, footprints: Footprints) -> "ComponentCounter":
        self.component_count_ = footprints.sizes[self.params.component_axis]

        return self

    def transform_one(self, frame: xr.DataArray) -> None:
        payload = {
            "type_": "component_count",
            "index": frame.coords[Axis.frame_coordinates].item(),
            "count": self.component_count_,
        }

        send_through(payload)
