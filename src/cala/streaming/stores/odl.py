from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from cala.streaming.core.stores import BaseStore, BodegaStore
from cala.streaming.types.odl import PixelStats, ComponentStats, Residual


# pixels x components
@dataclass(kw_only=True)
class PixelStatsStore(BodegaStore):
    spatial_axes: Tuple[str, ...]
    """The spatial axes of the footprints."""

    @property
    def data_type(self):
        return PixelStats

    def temporal_update(
        self, last_streamed_data: DataArray, ids: List[str]
    ) -> None: ...


# components x components
@dataclass(kw_only=True)
class ComponentStatsStore(BodegaStore):
    @property
    def data_type(self):
        return ComponentStats

    def temporal_update(
        self, last_streamed_data: DataArray, ids: List[str]
    ) -> None: ...


# this doesn't technically need a store. no association with components
@dataclass(kw_only=True)
class ResidualStore(BaseStore):
    spatial_axes: Tuple[str, ...]

    @property
    def data_type(self):
        return Residual

    def temporal_update(
        self, last_streamed_data: DataArray, ids: List[str]
    ) -> None: ...

    def generate_warehouse(self, data_array: np.ndarray | xr.DataArray) -> Residual:
        return Residual(data_array, dims=self.dimensions)
