from dataclasses import dataclass
from typing import Type, Optional, get_origin, Annotated

import xarray as xr

from cala.streaming.core import ObservableStore


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
            Optional[ObservableStore]: The requested Observable instance if found, None otherwise.
        """
        store_type = self._get_store_type(type_)
        if store_type is None:
            return
        for attr_name, attr_type in self.__annotations__.items():
            if attr_type == store_type:
                return getattr(self, attr_name)

    def init(self, result: xr.DataArray, type_: Type) -> None:
        """Store one or more DataArray results in their appropriate Observable containers.

        This method automatically determines the correct storage location based on the
        type of the input DataArray(s).

        Args:
            result: A single xr.DataArray to be stored. Must correspond to a valid Observable type.
            type_: type of the result. If an observable, should be an Annotated type that links to Store class.
        """
        target_store_type = self._get_store_type(type_)
        if target_store_type is None:
            return

        store_name = target_store_type.__name__.lower()
        # Add to annotations
        self.__annotations__[store_name] = target_store_type
        # Create and set the store
        setattr(self, store_name, result)

    @staticmethod
    def _get_store_type(type_: Type) -> type | None:
        if get_origin(type_) is Annotated:
            if issubclass(type_.__metadata__[0], ObservableStore):
                return type_.__metadata__[0]
