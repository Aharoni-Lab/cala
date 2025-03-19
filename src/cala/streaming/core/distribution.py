from dataclasses import dataclass, field
from typing import Type, Optional

import xarray as xr

from cala.streaming.core import Observable, Footprints, Traces
from cala.streaming.stores.odl import PixelStats, ComponentStats, Residual


@dataclass
class Distributor:
    """Manages a collection of fluorescent components (neurons and background).

    This class serves as a central manager for different storages,
    including spatial footprints, temporal traces, and various statistics.
    """

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    id_coord: str = "id_"
    """Name of the coordinate used to identify individual components with unique IDs."""
    type_coord: str = "type_"
    """Name of the coordinate used to specify component types (e.g., neuron, background)."""

    footprints: Footprints = field(default_factory=Footprints)
    """Storage for spatial footprints of components, representing their locations and shapes."""
    traces: Traces = field(default_factory=Traces)
    """Storage for temporal traces, containing the time-varying activity of each component."""

    pixel_stats: PixelStats = field(default_factory=PixelStats)
    """Storage for pixel-level statistics computed across the field of view."""
    component_stats: ComponentStats = field(default_factory=ComponentStats)
    """Storage for component-level statistics computed for each identified component."""
    residual: Residual = field(default_factory=Residual)
    """Storage for residual signals remaining after component extraction."""

    def get(self, type_: Type) -> Optional[Observable]:
        """Retrieve a specific Observable instance based on its type.

        Args:
            type_ (Type): The type of Observable to retrieve (e.g., Footprints, Traces).

        Returns:
            Optional[Observable]: The requested Observable instance if found, None otherwise.
        """
        for attr_name, attr_type in self.__annotations__.items():
            if issubclass(attr_type, Observable) and attr_type == type_:
                return getattr(self, attr_name)

    def collect(self, result: xr.DataArray | tuple[xr.DataArray, ...]) -> None:
        """Store one or more DataArray results in their appropriate Observable containers.

        This method automatically determines the correct storage location based on the
        type of the input DataArray(s).

        Args:
            result: Either a single xr.DataArray or a tuple of DataArrays to be stored.
                    Each DataArray must correspond to a valid Observable type.
        """
        results = (result,) if isinstance(result, xr.DataArray) else result

        for result in results:
            # determine which store to input the value into
            for attr_name, attr_type in self.__annotations__.items():
                if issubclass(attr_type, Observable) and isinstance(result, attr_type):
                    setattr(self, attr_name, result)
