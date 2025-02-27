from typing import List, Optional, Set

import numpy as np
from scipy import sparse

from .base import FluorescentObject


class ComponentManager:
    """Manages a collection of fluorescent components (neurons and background)."""

    def __init__(self):
        self._components: List[FluorescentObject] = list()
        self._footprint_shape: Optional[tuple] = None
        self._n_frames: Optional[int] = None

    @property
    def footprints(self) -> np.ndarray:
        """Returns concatenated footprints as a sparse 3D array (n_components, height, width)."""
        if not self._components:
            return np.array([np.array((0, 0))] * 3)
        return np.array(
            [component.footprint.toarray() for component in self._components]
        )

    @property
    def time_traces(self) -> np.ndarray:
        """Returns concatenated time traces as a 2D array (n_components, time).
        WARNING: This loads all time traces into memory at once. For large datasets,
        use get_time_traces_batch() or iterate_time_traces() instead.
        """
        if not self._components:
            return np.array([])
        return np.stack(
            [np.array(component.time_trace) for component in self._components]
        )

    def get_time_traces_batch(self, start_time: int, end_time: int) -> np.ndarray:
        """Get a batch of time traces for all components.

        Args:
            start_time: Start time index (inclusive)
            end_time: End time index (exclusive)

        Returns:
            2D array of shape (n_components, batch_time) with time traces
        """
        if not self._components:
            return np.array([])
        return np.stack(
            [
                np.array(component.time_trace[start_time:end_time])
                for component in self._components
            ]
        )

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
        first_component = next(iter(self._components))
        total_time = len(first_component.time_trace)

        for start_idx in range(0, total_time, batch_size):
            end_idx = min(start_idx + batch_size, total_time)
            yield start_idx, end_idx, self.get_time_traces_batch(start_idx, end_idx)

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return len(self._components)

    @property
    def neuron_indices(self) -> List[int]:
        """Returns a list of neuron indices."""
        return [
            idx
            for idx, component in enumerate(self._components)
            if component.__class__.__name__ == "Neuron"
        ]

    @property
    def background_indices(self) -> List[int]:
        """Returns a list of background indices."""
        return [
            idx
            for idx, component in enumerate(self._components)
            if component.__class__.__name__ == "Background"
        ]

    def get_component(self, component_idx: int) -> Optional[FluorescentObject]:
        """
        Get a component by its ID.

        Args:
            component_id: Unique identifier of the component

        Returns:
            Component if found, None otherwise
        """
        try:
            return self._components[component_idx]
        except IndexError:
            return None

    def add_component(self, component: FluorescentObject) -> None:
        """
        Add a new component while validating shape consistency.

        Args:
            component: FluorescentObject to add

        Raises:
            ValueError: If component dimensions don't match existing components
        """
        if not self._components:
            self._footprint_shape = component.footprint.shape
            self._n_frames = len(component.time_trace)
        else:
            if component.footprint.shape != self._footprint_shape:
                raise ValueError(
                    f"Component footprint shape {component.footprint.shape} "
                    f"doesn't match expected shape {self._footprint_shape}"
                )
            if len(component.time_trace) != self._n_frames:
                raise ValueError(
                    f"Component time trace length {len(component.time_trace)} "
                    f"doesn't match expected length {self._n_frames}"
                )

        # Update overlapping objects
        self._update_overlaps(component)
        self._components.append(component)

    def remove_component(self, component_idx: int) -> Optional[FluorescentObject]:
        """
        Remove a component by its ID and update overlapping relationships.

        Args:
            component_id: ID of component to remove

        Returns:
            Removed component if found, None otherwise
        """
        component = self._components.pop(component_idx)
        if component is None:
            return None

        # Remove this component from others' overlapping lists
        for other in self._components:
            other.overlapping_objects.discard(component)

        return component

    def update_component_timetrace(self, component_idx: int, time_trace: np.ndarray):
        """
        Update a component time trace.

        Args:
            component_id: ID of component to update
            time_trace: New time trace of the component

        Returns:
            True if component was found and updated, False otherwise
        """
        component = self._components[component_idx]

        if component is None:
            return False

        component.time_trace = time_trace
        return True

    def update_component_footprint(
        self, component_idx: int, footprint: np.ndarray
    ) -> bool:
        """
        Update a component footprint.

        Args:
            component_id: ID of component to update
            footprint: New footprint of the component

        Returns:
            True if component was found and updated, False otherwise

        Raises:
            ValueError: If new component dimensions don't match
        """
        component = self._components[component_idx]
        if component is None:
            return False

        # Validate shapes
        if footprint.shape != self._footprint_shape:
            raise ValueError("New footprint shape doesn't match")

        # Remove old footprint from others' overlapping lists
        for other in self._components:
            other.overlapping_objects.discard(component)
        component.overlapping_objects = set()

        component.footprint = footprint

        # Update overlaps for new footprint
        self._update_overlaps(component)
        return True

    def _update_overlaps(self, new_component: FluorescentObject) -> None:
        """Update overlapping relationships for a new component."""
        new_footprint_nonzero = new_component.footprint > 0
        for existing in self._components:
            # Use efficient sparse matrix operations
            if self._check_overlap(new_footprint_nonzero, existing.footprint > 0):
                new_component.overlapping_objects.add(existing)
                existing.overlapping_objects.add(new_component)

    @staticmethod
    def _check_overlap(
        footprint1: sparse.csr_matrix, footprint2: sparse.csr_matrix
    ) -> bool:
        """Check if two footprints overlap using efficient sparse operations."""
        # Element-wise multiplication will be sparse and fast
        overlap = footprint1.multiply(footprint2)
        return overlap.nnz > 0  # Check if any non-zero elements exist

    def get_components_by_type(self, component_type: type) -> List[FluorescentObject]:
        """
        Get all components of a specific type.

        Args:
            component_type: Type to filter by (Neuron or Background)

        Returns:
            List of components matching the type
        """
        return [
            component
            for component in self._components
            if isinstance(component, component_type)
        ]

    def get_overlapping_components(
        self, component_idx: int
    ) -> Optional[Set[FluorescentObject]]:
        """
        Get all components overlapping with the component with given ID.

        Args:
            component_id: ID of component to find overlaps for

        Returns:
            Set of overlapping components if component found, None otherwise
        """
        component = self._components[component_idx]
        return component.overlapping_objects if component else None
