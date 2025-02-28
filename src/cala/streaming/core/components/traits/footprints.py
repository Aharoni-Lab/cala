from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import xarray as xr


@dataclass
class FootprintManager:
    """Manages spatial footprints and their relationships."""

    component_axis: str = "component"
    """The axis of the component."""
    spatial_axes: Tuple[str, ...] = ("width", "height")
    """The spatial axes of the footprints."""

    _footprints: xr.DataArray = field(init=False, repr=False)
    """The footprints of the components. Shape: (n_components, *spatial_axes)."""
    _overlapping_components: Dict[int, Set[int]] = field(default_factory=dict)
    """Maps component IDs to sets of overlapping component IDs."""

    @property
    def footprints(self) -> xr.DataArray:
        """Returns the footprints as an xarray DataArray."""
        return self._footprints

    @property
    def footprints_dimensions(self) -> Tuple[str, ...]:
        """Returns the dimensions of the footprints."""
        return (self.component_axis, *self.spatial_axes)

    def initialize(self, footprints: xr.DataArray) -> None:
        """Initialize footprints data structure."""
        if set(footprints.dims) != set(self.footprints_dimensions):
            raise ValueError(
                f"Footprints dimensions must be {self.footprints_dimensions}"
            )
        self._footprints = footprints

    def add_footprint(self, component_id: int, footprint: xr.DataArray) -> None:
        """Add a new footprint."""
        # Expand and assign coordinates for the component axis
        footprint_expanded = footprint.expand_dims(self.component_axis).assign_coords(
            {self.component_axis: [component_id]}
        )

        if not hasattr(self, "_footprints"):
            self._footprints = footprint_expanded
        else:
            self._footprints = xr.concat(
                [self._footprints, footprint_expanded],
                dim=self.component_axis,
            )

        # Initialize overlap tracking and update overlaps
        self._overlapping_components[component_id] = set()
        self._update_overlaps(component_id)

    def remove_footprint(self, component_id: int) -> None:
        """Remove a footprint."""
        self._footprints = self._footprints.drop_sel(
            {self.component_axis: component_id}
        )

        # Clean up overlap tracking
        self._overlapping_components.pop(component_id, None)
        for overlaps in self._overlapping_components.values():
            overlaps.discard(component_id)

    def update_footprint(self, component_id: int, footprint: xr.DataArray) -> None:
        """Update an existing footprint."""
        # Create a condition mask for the component we want to update
        condition = self._footprints[self.component_axis] == component_id
        # Use where to update only the values for this component
        self._footprints = xr.where(condition, footprint, self._footprints)

        # Reset and recompute overlaps
        self._overlapping_components[component_id] = set()
        self._update_overlaps(component_id)

    def get_overlapping_components(self, component_id: int) -> Optional[Set[int]]:
        """Get all components overlapping with the given component."""
        return self._overlapping_components.get(component_id)

    def _update_overlaps(self, new_component_id: int) -> None:
        """Update overlapping relationships for a component."""
        # Get boolean mask for new component
        new_footprint_mask = (
            self._footprints.loc[{self.component_axis: new_component_id}] > 0
        )

        for existing_id in set(self._footprints.coords[self.component_axis].values):
            if existing_id == new_component_id:
                continue

            # Initialize overlapping set for existing component if needed
            if existing_id not in self._overlapping_components:
                self._overlapping_components[existing_id] = set()

            # Get boolean mask for existing component
            existing_footprint_mask = (
                self._footprints.loc[{self.component_axis: existing_id}] > 0
            )

            # Check overlap using xarray operations
            if bool((new_footprint_mask & existing_footprint_mask).any()):
                self._overlapping_components[new_component_id].add(existing_id)
                self._overlapping_components[existing_id].add(new_component_id)
