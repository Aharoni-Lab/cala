from typing import List, Optional

import numpy as np

from .base import FluorescentObject


class Manager:
    """Manages a collection of fluorescent components (neurons and background) for analysis."""

    def __init__(self):
        self.components: List[FluorescentObject] = []
        self._footprint_shape: Optional[tuple] = None
        self._n_timepoints: Optional[int] = None

    @property
    def footprints(self) -> np.ndarray:
        """Returns concatenated footprints as a 3D array (n_components, height, width)."""
        if not self.components:
            return np.array([])
        return np.stack([component.footprint for component in self.components])

    @property
    def time_traces(self) -> np.ndarray:
        """Returns concatenated time traces as a 2D array (n_components, time)."""
        if not self.components:
            return np.array([])
        return np.stack([component.time_trace for component in self.components])

    def add_component(self, component: FluorescentObject) -> None:
        """
        Add a new component while validating shape consistency.

        Args:
            component: FluorescentObject to add

        Raises:
            ValueError: If component dimensions don't match existing components
        """
        if not self.components:
            self._footprint_shape = component.footprint.shape
            self._n_timepoints = len(component.time_trace)
        else:
            if component.footprint.shape != self._footprint_shape:
                raise ValueError(
                    f"Component footprint shape {component.footprint.shape} "
                    f"doesn't match expected shape {self._footprint_shape}"
                )
            if len(component.time_trace) != self._n_timepoints:
                raise ValueError(
                    f"Component time trace length {len(component.time_trace)} "
                    f"doesn't match expected length {self._n_timepoints}"
                )

        # Update overlapping objects
        self._update_overlaps(component)
        self.components.append(component)

    def remove_component(self, index: int) -> FluorescentObject:
        """
        Remove a component and update overlapping relationships.

        Args:
            index: Index of component to remove

        Returns:
            Removed component
        """
        if not 0 <= index < len(self.components):
            raise IndexError("Component index out of range")

        component = self.components.pop(index)

        # Remove this component from others' overlapping lists
        for other in self.components:
            if component in other.overlapping_objects:
                other.overlapping_objects.remove(component)

        return component

    def update_component(self, index: int, new_component: FluorescentObject) -> None:
        """
        Update a component while maintaining consistency.

        Args:
            index: Index of component to update
            new_component: New component to replace with

        Raises:
            ValueError: If new component dimensions don't match
        """
        if not 0 <= index < len(self.components):
            raise IndexError("Component index out of range")

        old_component = self.components[index]

        # Validate shapes
        if new_component.footprint.shape != self._footprint_shape:
            raise ValueError("New component footprint shape doesn't match")
        if len(new_component.time_trace) != self._n_timepoints:
            raise ValueError("New component time trace length doesn't match")

        # Remove old component from others' overlapping lists
        for other in self.components:
            if old_component in other.overlapping_objects:
                other.overlapping_objects.remove(old_component)

        # Update overlaps for new component
        self._update_overlaps(new_component)
        self.components[index] = new_component

    def _update_overlaps(self, new_component: FluorescentObject) -> None:
        """Update overlapping relationships for a new component."""
        for existing in self.components:
            if self._check_overlap(new_component.footprint, existing.footprint):
                new_component.overlapping_objects.append(existing)
                existing.overlapping_objects.append(new_component)

    @staticmethod
    def _check_overlap(footprint1: np.ndarray, footprint2: np.ndarray) -> bool:
        """Check if two footprints overlap."""
        # Consider overlap if any pixel is non-zero in both footprints
        return np.any((footprint1 > 0) & (footprint2 > 0))

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
            for component in self.components
            if isinstance(component, component_type)
        ]

    def get_overlapping_components(self, index: int) -> List[FluorescentObject]:
        """
        Get all components overlapping with the component at given index.

        Args:
            index: Index of component to find overlaps for

        Returns:
            List of overlapping components
        """
        if not 0 <= index < len(self.components):
            raise IndexError("Component index out of range")
        return self.components[index].overlapping_objects
