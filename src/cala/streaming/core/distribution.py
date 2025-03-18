from dataclasses import dataclass, field
from typing import Type, Optional

import xarray as xr

from cala.streaming.core import Observable, Footprints, Traces
from cala.streaming.stores.odl import PixelStats, ComponentStats, Residual


@dataclass
class Distributor:
    """Manages a collection of fluorescent components (neurons and background)."""

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    id_coord: str = "id_"
    type_coord: str = "type_"

    footprints: Footprints = field(default_factory=Footprints)
    traces: Traces = field(default_factory=Traces)

    pixel_stats: PixelStats = field(default_factory=PixelStats)
    component_stats: ComponentStats = field(default_factory=ComponentStats)
    residual: Residual = field(default_factory=Residual)

    def get(self, type_: Type) -> Optional[Observable]:

        for attr_name, attr_type in self.__annotations__.items():
            if issubclass(attr_type, Observable) and attr_type == type_:
                return getattr(self, attr_name)

    def collect(self, result: xr.DataArray | tuple[xr.DataArray, ...]) -> None:
        results = (result,) if isinstance(result, xr.DataArray) else result

        for result in results:
            # determine which store to input the value into
            for attr_name, attr_type in self.__annotations__.items():
                if issubclass(attr_type, Observable) and isinstance(result, attr_type):
                    setattr(self, attr_name, result)
