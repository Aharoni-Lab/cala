from dataclasses import dataclass, field
from typing import Tuple, List

import xarray as xr

from cala.streaming.core.components.stores import BaseStore
from cala.streaming.types.types import Footprints


@dataclass
class FootprintStore(BaseStore):
    """Manages spatial footprints and their relationships.

    assign (.array) just refreshes the entire thing. (whole ✅)
    replace swaps in existing footprints. (one ✅ & batch ✅)
    update does the same thing as replace ✅
    insert concatenates new footprints. (one ✅ & batch ✅)
    """

    component_axis: str = "component"
    """The axis of the component."""
    spatial_axes: Tuple[str, ...] = ("width", "height")
    """The spatial axes of the footprints."""

    _footprints: Footprints = field(init=False, repr=False)
    """The footprints of the components. Shape: (n_components, *spatial_axes)."""

    @property
    def array(self) -> Footprints:
        """Returns the footprints as an xarray DataArray."""
        return self._footprints

    @array.setter
    def array(self, value: Footprints) -> None:
        if set(value.dims) != set(self.dimensions):
            raise ValueError(f"Footprints dimensions must be {self.dimensions}")
        self._footprints = Footprints(value)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        """Returns the dimensions of the footprints."""
        return (self.component_axis, *self.spatial_axes)

    def insert(self, footprints: Footprints) -> None:
        """Insert a new footprint."""
        if not hasattr(self, "_footprints"):
            self._footprints = Footprints(footprints)
        else:
            self._footprints = Footprints(
                xr.concat(
                    [self._footprints, footprints],
                    dim=self.component_axis,
                )
            )

    def remove(self, component_ids: List[int]) -> None:
        """Remove a footprint."""
        self._footprints = self._footprints.drop_sel(
            {self.component_axis: component_ids}
        )

    def replace(self, footprints: Footprints) -> None:
        """Replace existing footprints."""
        # Create a condition mask for the component we want to update
        condition = (
            self._footprints[self.component_axis]
            == footprints.coords[self.component_axis]
        )
        # Use where to update only the values for this component
        self._footprints = Footprints(xr.where(condition, footprints, self._footprints))

    def update(self, footprints: Footprints) -> None:
        self.replace(footprints)
