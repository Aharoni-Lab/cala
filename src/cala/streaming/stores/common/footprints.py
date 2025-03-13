from dataclasses import dataclass
from typing import Tuple, List, Dict, Type

import xarray as xr

from cala.streaming.core import BaseStore
from cala.streaming.types import Footprints


@dataclass(kw_only=True)
class FootprintStore(BaseStore):
    """Manages spatial footprints and their relationships."""

    spatial_axes: Tuple[str, ...]
    """The spatial axes of the footprints."""

    @property
    def data_type(self) -> Type:
        return Footprints

    def temporal_update(self, last_streamed_data: xr.DataArray, ids: List[str]) -> None:
        pass

    def crop(self, to_drop: Dict[str, List[int]]) -> xr.DataArray:
        return self._warehouse.drop_sel(to_drop)
