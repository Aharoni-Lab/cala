from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Set, Type

import xarray as xr

from cala.streaming.core.components.traits import FootprintManager, TraceManager
from cala.streaming.core.components.types import FluorescentObject
from .registry import ComponentRegistry


@dataclass
class ComponentManager:
    """Manages a collection of fluorescent components (neurons and background)."""

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    _registry: ComponentRegistry = field(default_factory=ComponentRegistry)
    _footprints: FootprintManager = field(default_factory=lambda: FootprintManager())
    _traces: TraceManager = field(default_factory=lambda: TraceManager())

    def __post_init__(self):
        # Ensure consistent axis names across managers
        self._footprints.component_axis = self.component_axis
        self._footprints.spatial_axes = self.spatial_axes
        self._traces.component_axis = self.component_axis
        self._traces.frame_axis = self.frame_axis

    @property
    def footprints(self) -> xr.DataArray:
        """Returns the footprints as an xarray DataArray."""
        return self._footprints.footprints

    @property
    def traces(self) -> xr.DataArray:
        """Returns the traces as an xarray DataArray."""
        return self._traces.traces

    @cached_property
    def footprints_dimensions(self) -> tuple:
        """Returns the dimensions of the footprints."""
        return self.component_axis, *self.spatial_axes

    @cached_property
    def traces_dimensions(self) -> tuple:
        """Returns the dimensions of the traces."""
        return self.component_axis, self.frame_axis

    @property
    def component_ids(self) -> Set[int]:
        """Returns all component IDs."""
        return self._registry.component_ids

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return self._registry.n_components

    @property
    def neuron_ids(self) -> Set[int]:
        """Returns a list of neuron IDs."""
        return {
            component_id
            for component_id in self.component_ids
            if self._registry.get(component_id).__class__.__name__ == "Neuron"
        }

    @property
    def background_ids(self) -> Set[int]:
        """Returns a list of background IDs."""
        return {
            component_id
            for component_id in self.component_ids
            if self._registry.get(component_id).__class__.__name__ == "Background"
        }

    def verify_component_consistency(self) -> None:
        """Verify that components are consistent across all managers."""
        footprint_ids = set(self.footprints.coords[self.component_axis].values)
        trace_ids = set(self.traces.coords[self.component_axis].values)
        registry_ids = self.component_ids

        if footprint_ids != trace_ids:
            raise ValueError(
                "Component IDs in footprints and traces must match. "
                f"Footprint IDs: {footprint_ids} "
                f"Trace IDs: {trace_ids}"
            )
        if footprint_ids != registry_ids:
            raise ValueError(
                "Component IDs in footprints must match registry IDs. "
                f"Footprint IDs: {footprint_ids} "
                f"Registry IDs: {registry_ids}"
            )

    def populate_from_footprints(
        self,
        footprints: xr.DataArray,
        component_type: Type[FluorescentObject],
    ) -> None:
        """Populate the component manager from footprints."""
        if set(footprints.dims) != set(self._footprints.footprints_dimensions):
            raise ValueError(
                f"Footprints dimensions must be {self._footprints.footprints_dimensions}"
            )

        if len(self._registry.component_ids) == 0:
            # Initialize from scratch
            components = [
                component_type() for _ in footprints.coords[self.component_axis]
            ]
            for component in components:
                self._registry.add(component)

            self._footprints.initialize(
                footprints.assign_coords(
                    {self.component_axis: list(self.component_ids)}
                )
            )
            self._traces.initialize(list(self.component_ids))
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
                self._footprints.update_footprint(
                    component_id,
                    existing_footprints.sel({self.component_axis: component_id}),
                )

        # Add new components
        if new_ids:
            new_components = [component_type() for _ in range(len(new_ids))]
            new_footprints = footprints.sel({self.component_axis: list(new_ids)})

            for component, component_id in zip(new_components, new_ids):
                self._registry.add(component)
                self._footprints.add_footprint(
                    component.id,
                    new_footprints.sel({self.component_axis: component_id}),
                )
                self._traces.add_trace(component.id)

    def populate_from_traces(self, traces: xr.DataArray) -> None:
        """Populate traces for existing components."""
        if set(traces.coords[self.component_axis].values) - self.component_ids:
            raise ValueError(
                "Cannot add traces for components that don't exist. "
                f"Unknown components: {set(traces.coords[self.component_axis].values) - self.component_ids}"
            )
        self._traces.append_frames(traces)

    def get_component(self, component_id: int) -> Optional[FluorescentObject]:
        """Get a component by its ID."""
        return self._registry.get(component_id)

    def get_components_by_type(
        self, component_type: Type[FluorescentObject]
    ) -> List[int]:
        """Get all component IDs of a specific type."""
        return self._registry.get_by_type(component_type)

    def get_overlapping_components(self, component_id: int) -> Optional[Set[int]]:
        """Get all components overlapping with the given component."""
        return self._footprints.get_overlapping_components(component_id)

    def add_component(
        self,
        component: FluorescentObject,
        footprint: xr.DataArray,
        time_trace: Optional[xr.DataArray] = None,
    ) -> None:
        """Add a new component with its data."""
        self._registry.add(component)
        self._footprints.add_footprint(component.id, footprint)
        self._traces.add_trace(component.id, time_trace)

    def remove_component(self, component_id: int) -> Optional[FluorescentObject]:
        """Remove a component and its data."""
        component = self._registry.remove(component_id)
        if component is not None:
            self._footprints.remove_footprint(component_id)
            self._traces.remove_trace(component_id)
        return component

    def update_component_timetrace(
        self, component_id: int, time_trace: xr.DataArray
    ) -> bool:
        """Update a component's time trace."""
        if component_id not in self.component_ids:
            return False
        self._traces.update_trace(component_id, time_trace)
        return True

    def update_component_footprint(
        self, component_id: int, footprint: xr.DataArray
    ) -> bool:
        """Update a component's footprint."""
        if component_id not in self.component_ids:
            return False
        self._footprints.update_footprint(component_id, footprint)
        return True

    def get_time_traces_batch(self, start_time: int, end_time: int) -> xr.DataArray:
        """Get a batch of time traces for all components."""
        return self._traces.get_batch(start_time, end_time)

    def iterate_time_traces(self, batch_size: int = 1000):
        """Iterate over time traces in batches."""
        yield from self._traces.iterate_batches(batch_size)
