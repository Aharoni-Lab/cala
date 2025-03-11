from dataclasses import dataclass
from typing import Tuple, List, Dict

import xarray as xr

from cala.streaming.core.stores import BaseStore


@dataclass
class FootprintStore(BaseStore):
    """Manages spatial footprints and their relationships."""

    spatial_axes: Tuple[str, ...] = ("width", "height")
    """The spatial axes of the footprints."""

    @property
    def dims(self) -> Tuple[str, ...]:
        return self.component_dim, *self.spatial_axes

    def temporal_update(self, last_streamed_data: xr.DataArray, ids: List[str]) -> None:
        pass

    def crop(self, to_drop: Dict[str, List[int]]) -> xr.DataArray:
        return self._warehouse.drop_sel(to_drop)
