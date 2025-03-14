from dataclasses import dataclass
from typing import List, Type

import xarray as xr

from cala.streaming.core.stores import UltimateStore
from cala.streaming.types import Traces


@dataclass(kw_only=True)
class TraceStore(UltimateStore):
    """Manages temporal traces for components."""

    frame_axis: str
    """The axis of the frames."""

    @property
    def data_type(self) -> Type:
        return Traces

    def temporal_update(
        self, last_streamed_data: xr.DataArray, ids: List[str]
    ) -> None: ...
