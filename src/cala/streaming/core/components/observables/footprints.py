from dataclasses import dataclass, field
from typing import Tuple

import xarray as xr

from cala.streaming.core import Footprints, Footprint


@dataclass
class FootprintStore:
    """Manages spatial footprints and their relationships."""

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
        self._footprints = Footprints(value)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        """Returns the dimensions of the footprints."""
        return (self.component_axis, *self.spatial_axes)

    def initialize(self, footprints: Footprints) -> None:
        """Initialize footprints data structure."""
        if set(footprints.dims) != set(self.dimensions):
            raise ValueError(f"Footprints dimensions must be {self.dimensions}")
        self._footprints = Footprints(footprints)

    def add(self, component_id: int, footprint: Footprint) -> None:
        """Add a new footprint."""
        # Expand and assign coordinates for the component axis
        footprint_expanded = footprint.expand_dims(self.component_axis).assign_coords(
            {self.component_axis: [component_id]}
        )

        if not hasattr(self, "_footprints"):
            self._footprints = Footprints(footprint_expanded)
        else:
            self._footprints = Footprints(
                xr.concat(
                    [self._footprints, footprint_expanded],
                    dim=self.component_axis,
                )
            )

    def remove(self, component_id: int) -> None:
        """Remove a footprint."""
        self._footprints = self._footprints.drop_sel(
            {self.component_axis: component_id}
        )

    def update(self, component_id: int, footprint: Footprint) -> None:
        """Update an existing footprint."""
        # Create a condition mask for the component we want to update
        condition = self._footprints[self.component_axis] == component_id
        # Use where to update only the values for this component
        self._footprints = Footprints(xr.where(condition, footprint, self._footprints))
