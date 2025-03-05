from dataclasses import dataclass, field
from typing import Optional, Set

import xarray as xr

from cala.streaming.core.components.observables import FootprintStore, TraceStore
from cala.streaming.core.components.categories import FluorescentObject
from .registry import Registry


@dataclass
class StoreManager:
    """Manages a collection of fluorescent components (neurons and background)."""

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    registry: Registry = field(default_factory=Registry)
    footprints: FootprintStore = field(default_factory=lambda: FootprintStore())
    traces: TraceStore = field(default_factory=lambda: TraceStore())

    def __post_init__(self):
        # Ensure consistent axis names across managers
        self.footprints.component_axis = self.component_axis
        self.footprints.spatial_axes = self.spatial_axes
        self.traces.component_axis = self.component_axis
        self.traces.frame_axis = self.frame_axis

    @property
    def neuron_ids(self) -> Set[int]:
        """Returns a list of neuron IDs."""
        return {
            component_id
            for component_id in self.registry.ids
            if self.registry.get(component_id).__class__.__name__ == "Neuron"
        }

    @property
    def background_ids(self) -> Set[int]:
        """Returns a list of background IDs."""
        return {
            component_id
            for component_id in self.registry.ids
            if self.registry.get(component_id).__class__.__name__ == "Background"
        }

    def populate_from_footprints(
        self,
        footprints: xr.DataArray,
    ) -> None:
        """Populate the component manager from footprints.

        Args:
            footprints: The footprints to populate from.
        """
        if set(footprints.dims) != set(self.footprints.dimensions):
            raise ValueError(
                f"Footprints dimensions must be {self.footprints.dimensions}"
            )

        component_class = self._component_type_map[component_type]

        if len(self.registry.ids) == 0:
            # Initialize from scratch
            components = [
                component_class() for _ in footprints.coords[self.component_axis]
            ]
            for component in components:
                self.registry.add(component)

            self.footprints.initialize(
                footprints.assign_coords(
                    {self.component_axis: list(self.component_ids)}
                )
            )
            self.traces.initialize(list(self.component_ids))
            return

        # Handle updates and additions
        input_ids = set(footprints.coords[self.component_axis].values)
        existing_ids = self.component_ids
        new_ids = input_ids - existing_ids
        existing_input_ids = input_ids & existing_ids

        # Update existing footprints
        if existing_input_ids:
            existing_footprints = footprints.sel(
                {self.component_axis: list(existing_input_ids)}
            )
            for component_id in existing_input_ids:
                self.footprints.update_footprint(
                    component_id,
                    existing_footprints.sel({self.component_axis: component_id}),
                )

        # Add new components
        if new_ids:
            new_components = [component_class() for _ in range(len(new_ids))]
            new_footprints = footprints.sel({self.component_axis: list(new_ids)})

            for component, component_id in zip(new_components, new_ids):
                self.registry.add(component)
                self.footprints.add_footprint(
                    component.id,
                    new_footprints.sel({self.component_axis: component_id}),
                )
                self.traces.add(component.id)

    def add_component(
        self,
        component: FluorescentObject,
        footprint: Footprint,
        time_trace: Optional[xr.DataArray] = None,
    ) -> None:
        """Add a new component with its data."""
        self.registry.add(component)
        self.footprints.add(component.id, footprint)
        self.traces.add(component.id, time_trace)

    def remove_component(self, component_id: int) -> Optional[FluorescentObject]:
        """Remove a component and its data."""
        component = self.registry.remove(component_id)
        if component is not None:
            self.footprints.remove(component_id)
            self.traces.remove(component_id)
        return component
