from dataclasses import dataclass

import xarray as xr

from cala.assets import Footprints
from cala.gui.nodes.util import send_through


@dataclass
class ComponentCounter:
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
