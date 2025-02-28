from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Set, Dict

import numpy as np
import xarray as xr

from cala.streaming.core.components.base import FluorescentObject


@dataclass
class ComponentManager:
    """Manages a collection of fluorescent components (neurons and background)."""

    component_axis: str = "component"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    _components: Dict[int, FluorescentObject] = field(default_factory=dict)
    """The components in the manager."""
    _footprint_shape: Optional[tuple] = None
    """The shape of the footprint."""
    _n_frames: Optional[int] = None
    """The number of frames."""

    _footprints: xr.DataArray = field(init=False, repr=False)
    """The footprints of the components. Shape: (n_components, *spatial_axes)."""
    _traces: xr.DataArray = field(init=False, repr=False)
    """The traces of the components. Shape: (n_components, n_frames)."""

    _overlapping_components: Dict[int, Set[int]] = field(default_factory=dict)
    """The overlapping components of the components. Key is component ID, value is set of overlapping component IDs."""

    @property
    def footprints(self) -> xr.DataArray:
        """Returns the footprints as an xarray DataArray."""
        return self._footprints

    @property
    def traces(self) -> xr.DataArray:
        """Returns the traces as an xarray DataArray."""
        return self._traces

    @cached_property
    def footprints_dimensions(self) -> tuple:
        """Returns the dimensions of the footprints."""
        return self.component_axis, *self.spatial_axes

    @cached_property
    def traces_dimensions(self) -> tuple:
        """Returns the dimensions of the traces."""
        return self.component_axis, self.frame_axis

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return len(self._components)

    @property
    def component_ids(self) -> Set[int]:
        return set(self._components.keys())

    @property
    def neuron_ids(self) -> Set[int]:
        """Returns a list of neuron IDs."""
        return {
            component.id
            for component in self._components.values()
            if component.__class__.__name__ == "Neuron"
        }

    @property
    def background_ids(self) -> Set[int]:
        """Returns a list of background IDs."""
        return {
            component.id
            for component in self._components.values()
            if component.__class__.__name__ == "Background"
        }

    def verify_component_consistency(self):
        if set(self.footprints.coords[self.component_axis].values()) != set(
            self.traces.coords[self.component_axis].values()
        ):
            raise ValueError(
                "Component IDs in footprints and traces must match. "
                f"Footprint IDs: {set(self.footprints.coords[self.component_axis].values())} "
                f"Traces IDs: {set(self.traces.coords[self.component_axis].values())}"
            )
        if (
            set(self.footprints.coords[self.component_axis].values())
            != self.component_ids
        ):
            raise ValueError(
                "Component IDs in footprints must match component IDs. "
                f"Footprint IDs: {set(self.footprints.coords[self.component_axis].values())} "
                f"Component IDs: {self.component_ids}"
            )

    def _initialize_components_from_footprints(
        self,
        footprints: xr.DataArray,
        component_type: type,
    ) -> None:
        """Initialize components and data structures from footprints when empty."""
        # Create components and map by their IDs
        components = [component_type() for _ in footprints.coords[self.component_axis]]
        self._components = {component.id: component for component in components}

        # Set footprints with component IDs as coordinates
        self._footprints = footprints.assign_coords(
            {self.component_axis: list(self.component_ids)}
        )

        # Create traces with matching component IDs
        self._traces = xr.DataArray(
            np.zeros((self.footprints.sizes[self.component_axis], 0)),
            coords={
                self.component_axis: list(self.component_ids),
                self.frame_axis: np.arange(0),
            },
            dims=self.traces_dimensions,
        )

    def _update_existing_footprints(
        self,
        footprints: xr.DataArray,
        existing_input_ids: set,
    ) -> None:
        """Update footprints for existing components."""
        if not existing_input_ids:
            return

        existing_footprints = footprints.sel(
            {self.component_axis: list(existing_input_ids)}
        )
        self._footprints.loc[{self.component_axis: list(existing_input_ids)}] = (
            existing_footprints
        )

    def _add_new_components(
        self,
        footprints: xr.DataArray,
        new_ids: set,
        component_type: type,
    ) -> None:
        """Add new components and their corresponding data."""
        if not new_ids:
            return

        # Create new components and map by their IDs
        new_components = [component_type() for _ in range(len(new_ids))]
        self._components.update(
            {component.id: component for component in new_components}
        )

        # Get and concatenate new footprints
        new_footprints = footprints.sel({self.component_axis: list(new_ids)})
        self._footprints = xr.concat(
            [
                self._footprints,
                new_footprints.assign_coords(
                    {self.component_axis: [c.id for c in new_components]}
                ),
            ],
            dim=self.component_axis,
        )

        # Add empty traces for new components
        new_traces = xr.DataArray(
            np.zeros((len(new_components), self._traces.sizes[self.frame_axis])),
            coords={
                self.component_axis: [c.id for c in new_components],
                self.frame_axis: self._traces.coords[self.frame_axis],
            },
            dims=self.traces_dimensions,
        )
        self._traces = xr.concat([self._traces, new_traces], dim=self.component_axis)

    def populate_from_footprints(
        self,
        footprints: xr.DataArray,
        component_type: type,
    ) -> None:
        """Populate the component manager from footprints.

        This method handles both initialization of an empty manager and updates to an
        existing one. For existing managers, it will:
        1. Update footprints for existing components
        2. Create new components for new footprints
        """
        if set(footprints.dims) != set(self.footprints_dimensions):
            raise ValueError(
                f"Footprints dimensions must be {self.footprints_dimensions}"
            )

        # Set axes to match footprints
        self.spatial_axes = tuple(footprints.dims)

        if len(self._components) == 0:
            self._initialize_components_from_footprints(footprints, component_type)
            return

        # Identify new vs existing components
        existing_ids = self.component_ids
        input_ids = set(footprints.coords[self.component_axis].values)
        new_ids = input_ids - existing_ids
        existing_input_ids = input_ids & existing_ids

        # Handle updates and additions
        self._update_existing_footprints(footprints, existing_input_ids)
        self._add_new_components(footprints, new_ids, component_type)

    def populate_from_traces(
        self,
        traces: xr.DataArray,
    ) -> None:
        """Populate the component manager from traces.

        This method appends new frames to all components. Components not explicitly
        provided in the input traces will have zeros for the new frames.
        """
        if set(traces.dims) != set(self.traces_dimensions):
            raise ValueError(f"Traces dimensions must be {self.traces_dimensions}")

        if set(traces.coords[self.component_axis].values) - self.component_ids:
            raise ValueError(
                f"There should be no new components, as knowing the trace without footprint is not possible. Unexpected components: {set(traces.coords[self.component_axis].values) - self.component_ids}"
            )

        # Create a zero-filled array for all components
        new_traces = xr.DataArray(
            np.zeros((len(self._components), traces.sizes[self.frame_axis])),
            coords={
                self.component_axis: list(self.component_ids),
                self.frame_axis: traces.coords[self.frame_axis],
            },
            dims=self.traces_dimensions,
        )

        # Update values for components that have explicit traces
        input_ids = set(traces.coords[self.component_axis].values)
        if input_ids:
            new_traces.loc[{self.component_axis: list(input_ids)}] = traces

        # Concatenate with existing traces
        self._traces = xr.concat([self._traces, new_traces], dim=self.frame_axis)

    def _update_existing_traces(
        self,
        traces: xr.DataArray,
        existing_input_ids: set,
    ) -> None:
        """Update traces for existing components."""
        if not existing_input_ids:
            return

        existing_traces = traces.sel({self.component_axis: list(existing_input_ids)})
        self._traces.loc[{self.component_axis: list(existing_input_ids)}] = (
            existing_traces
        )

    def get_time_traces_batch(self, start_time: int, end_time: int) -> xr.DataArray:
        """Get a batch of time traces for all components.

        Args:
            start_time: Start time index (inclusive)
            end_time: End time index (exclusive)

        Returns:
            2D array of shape (n_components, batch_time) with time traces
        """
        if not self._components:
            return xr.DataArray(np.array([]), dims=self.traces_dimensions)
        return self.traces.sel({self.frame_axis: slice(start_time, end_time)})

    def iterate_time_traces(self, batch_size: int = 1000):
        """Iterate over time traces in batches to avoid loading everything into memory.

        Args:
            batch_size: Number of time points to load at once

        Yields:
            Tuple of (start_time, end_time, batch_data) where batch_data is a
            2D array of shape (n_components, batch_size)
        """
        if not self._components:
            return

        # Get total time points from first component
        total_time = self.traces.sizes[self.frame_axis]

        for start_idx in range(0, total_time, batch_size):
            end_idx = min(start_idx + batch_size, total_time)
            yield start_idx, end_idx, self.get_time_traces_batch(start_idx, end_idx)

    def get_component(self, component_id: int) -> Optional[FluorescentObject]:
        """Get a component by its ID."""
        return self._components.get(component_id)

    def add_component(
        self,
        component: FluorescentObject,
        footprint: xr.DataArray,
        time_trace: xr.DataArray,
    ) -> None:
        """Add a new component while validating shape consistency."""
        self._components[component.id] = component

        # Add footprint and traces by appending to existing data. The component coordinate
        # is set to the new component ID.
        if not self._components:
            # Initialize arrays if this is the first component
            self._footprints = footprint.expand_dims(self.component_axis).assign_coords(
                {self.component_axis: [component.id]}
            )
            self._traces = time_trace.expand_dims(self.component_axis).assign_coords(
                {self.component_axis: [component.id]}
            )
        else:
            # Append to existing arrays
            self._footprints = xr.concat(
                [
                    self._footprints,
                    footprint.expand_dims(self.component_axis).assign_coords(
                        {self.component_axis: [component.id]}
                    ),
                ],
                dim=self.component_axis,
            )
            self._traces = xr.concat(
                [
                    self._traces,
                    time_trace.expand_dims(self.component_axis).assign_coords(
                        {self.component_axis: [component.id]}
                    ),
                ],
                dim=self.component_axis,
            )

        if self._components:
            # Update overlapping objects
            self._update_overlaps(component.id)

    def remove_component(self, component_id: int) -> Optional[FluorescentObject]:
        """Remove a component by its ID."""
        if component_id not in self._components:
            return None

        component = self._components.pop(component_id)

        # drop the components from footprint and traces
        self._footprints = self._footprints.drop_sel(
            {self.component_axis: component_id}
        )
        self._traces = self._traces.drop_sel({self.component_axis: component_id})

        # Remove this component from others' overlapping lists
        self._overlapping_components.pop(component_id)
        for value in self._overlapping_components.values():
            value.discard(component_id)

        return component

    def update_component_timetrace(self, component_id: int, time_trace: np.ndarray):
        """Update a component time trace."""
        if component_id not in self._components.keys():
            return False

        # Validate shapes
        if time_trace.shape != self._traces.shape[1:]:
            raise ValueError("New time trace shape doesn't match")

        self._traces.loc[{self.component_axis: component_id}] = time_trace
        return True

    def update_component_footprint(
        self, component_id: int, footprint: np.ndarray
    ) -> bool:
        """Update a component footprint."""
        if component_id not in self._components:
            return False

        # Validate shapes
        if footprint.shape != self._footprint_shape:
            raise ValueError("New footprint shape doesn't match")

        # Remove old footprint from others' overlapping lists
        for value in self._overlapping_components.values():
            value.discard(component_id)

        self._footprints.loc[{self.component_axis: component_id}] = footprint

        # Update overlaps for new footprint
        self._overlapping_components[component_id] = set()
        self._update_overlaps(component_id)
        return True

    def _update_overlaps(self, new_component_id: int) -> None:
        """Update overlapping relationships for a new component."""
        # Get boolean mask for new component
        new_footprint_mask = (
            self._footprints.loc[{self.component_axis: new_component_id}] > 0
        )

        # Initialize overlapping set for new component if needed
        if new_component_id not in self._overlapping_components:
            self._overlapping_components[new_component_id] = set()

        for existing_component_id in self._components.keys():
            if existing_component_id == new_component_id:
                continue

            # Initialize overlapping set for existing component if needed
            if existing_component_id not in self._overlapping_components:
                self._overlapping_components[existing_component_id] = set()

            # Get boolean mask for existing component
            existing_footprint_mask = (
                self._footprints.loc[{self.component_axis: existing_component_id}] > 0
            )

            # Check overlap using xarray operations and convert to Python boolean
            if bool((new_footprint_mask & existing_footprint_mask).any()):
                self._overlapping_components[new_component_id].add(
                    existing_component_id
                )
                self._overlapping_components[existing_component_id].add(
                    new_component_id
                )

    @staticmethod
    def _check_overlap(footprint1: xr.DataArray, footprint2: xr.DataArray) -> bool:
        """Check if two footprints overlap using xarray boolean operations."""
        return bool((footprint1 & footprint2).any())

    def get_components_by_type(self, component_type: type) -> List[int]:
        """
        Get all components IDs of a specific type.

        Args:
            component_type: Type to filter by (Neuron or Background)

        Returns:
            List of component IDs matching the type
        """
        return [
            component.id
            for component in self._components.values()
            if isinstance(component, component_type)
        ]

    def get_overlapping_components(self, component_id: int) -> Optional[Set[int]]:
        """
        Get all components overlapping with the component with given ID.

        Args:
            component_id: ID of component to find overlaps for

        Returns:
            Set of overlapping components if component found, None otherwise
        """
        return self._overlapping_components.get(component_id, None)
