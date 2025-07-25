from dataclasses import dataclass

import xarray as xr
from river.base import Transformer

from cala.gui.nodes.util import send_through
from cala.models import Params
from cala.stores.common import Footprints


@dataclass
class ComponentCounterParams(Params):
    pass

    def validate(self) -> None:
        pass


@dataclass
class ComponentCounter(Transformer):
    params: ComponentCounterParams
    component_count_: int = 0

    def learn_one(self, footprints: Footprints) -> "ComponentCounter":
        self.component_count_ = footprints.sizes[self.params.component_dim]

        return self

    def transform_one(self, frame: xr.DataArray) -> None:
        payload = {
            "type_": "component_count",
            "index": frame.coords[self.params.frame_coord].item(),
            "count": self.component_count_,
        }

        send_through(payload)
