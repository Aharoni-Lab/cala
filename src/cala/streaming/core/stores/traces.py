from dataclasses import dataclass
from typing import Tuple, List

import xarray as xr

from cala.streaming.core.stores import BaseStore


@dataclass(kw_only=True)
class TraceStore(BaseStore):
    """Manages temporal traces for components."""

    frame_axis: str
    """The axis of the frames."""

    @property
    def dims(self) -> Tuple[str, ...]:
        return self.frame_axis, self.component_dim

    def temporal_update(
        self, last_streamed_data: xr.DataArray, ids: List[str]
    ) -> None: ...
