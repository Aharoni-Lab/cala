from typing import Annotated

import xarray as xr

from cala.streaming.core import ObservableStore


class FootprintStore(ObservableStore):
    """Spatial footprints of identified components.

    Represents the spatial distribution patterns of components (neurons or background)
    in the field of view. Each footprint typically contains the spatial extent and
    intensity weights of a component.
    """

    pass


Footprints = Annotated[xr.DataArray, FootprintStore]


class TraceStore(ObservableStore):
    """Temporal activity traces of identified components.

    Contains the time-varying fluorescence signals of components across frames,
    representing their activity patterns over time.
    """

    def update(self, data: xr.DataArray) -> None:
        # either has frames or components axis.
        # are we making copies??
        self._warehouse = xr.concat(
            [self._warehouse, data],
            dim=(set(self._warehouse.dims) - set(data.dims)).pop(),
        )


Traces = Annotated[xr.DataArray, TraceStore]
