from functools import cached_property
from typing import List, Optional, Dict, Set

import numpy as np

from .base import FluorescentObject


class ComponentManager:
    """Manages a collection of fluorescent components (neurons and background)."""

    def __init__(self):
        self._components: Dict[int, FluorescentObject] = {}
        self._footprint_shape: Optional[tuple] = None
        self._n_timepoints: Optional[int] = None

    def get_component(self, component_id: int) -> Optional[FluorescentObject]:
        """
        Get a component by its ID.

        Args:
            component_id: Unique identifier of the component

        Returns:
            Component if found, None otherwise
        """
        return self._components.get(component_id)

    @cached_property
    def footprints(self) -> np.ndarray:
        """Returns concatenated footprints as a 3D array (n_components, height, width)."""
        if not self._components:
            return np.array([])
        return np.stack(
            [component.footprint for component in self._components.values()]
        )

    @cached_property
    def time_traces(self) -> np.ndarray:
        """Returns concatenated time traces as a 2D array (n_components, time)."""
        if not self._components:
            return np.array([])
        return np.stack(
            [component.time_trace for component in self._components.values()]
        )

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
        self._components[component._id] = component

    def remove_component(self, component_id: int) -> Optional[FluorescentObject]:
        """
        Remove a component by its ID and update overlapping relationships.

        Args:
            component_id: ID of component to remove

        Returns:
            Removed component if found, None otherwise
        """
        component = self._components.pop(component_id, None)
        if component is None:
            return None

        # Remove this component from others' overlapping lists
        for other in self._components.values():
            other.overlapping_objects.discard(component)

        return component

    def update_component(
        self, component_id: int, new_component: FluorescentObject
    ) -> bool:
        """
        Update a component while maintaining consistency.

        Args:
            component_id: ID of component to update
            new_component: New component to replace with

        Returns:
            True if component was found and updated, False otherwise

        Raises:
            ValueError: If new component dimensions don't match
        """
        old_component = self._components.get(component_id)
        if old_component is None:
            return False

        # Validate shapes
        if new_component.footprint.shape != self._footprint_shape:
            raise ValueError("New component footprint shape doesn't match")
        if len(new_component.time_trace) != self._n_timepoints:
            raise ValueError("New component time trace length doesn't match")

        # Remove old component from others' overlapping lists
        for other in self._components.values():
            other.overlapping_objects.discard(old_component)

        # Update overlaps for new component
        self._update_overlaps(new_component)
        self._components[component_id] = new_component
        return True

    def _update_overlaps(self, new_component: FluorescentObject) -> None:
        """Update overlapping relationships for a new component."""
        for existing in self._components.values():
            if self._check_overlap(new_component.footprint, existing.footprint):
                new_component.overlapping_objects.add(existing)
                existing.overlapping_objects.add(new_component)

    @staticmethod
    def _check_overlap(footprint1: np.ndarray, footprint2: np.ndarray) -> bool:
        """Check if two footprints overlap."""
        # Consider overlap if any pixel is non-zero in both footprints
        return bool(np.any((footprint1 > 0) & (footprint2 > 0)))

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
            for component in self._components.values()
            if isinstance(component, component_type)
        ]

    def get_overlapping_components(
        self, component_id: int
    ) -> Optional[Set[FluorescentObject]]:
        """
        Get all components overlapping with the component with given ID.

        Args:
            component_id: ID of component to find overlaps for

        Returns:
            Set of overlapping components if component found, None otherwise
        """
        component = self._components.get(component_id)
        return component.overlapping_objects if component else None
