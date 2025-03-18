from dataclasses import dataclass, field
from typing import Self, List, Set

import numpy as np
import sparse
import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters, Footprints, TransformerMeta
from cala.streaming.stores.odl import OverlapGroups


@dataclass
class OverlapGroupsInitializerParams(Parameters):
    """Parameters for computing spatially overlapping component groups.

    This class defines the configuration parameters needed for determining
    groups of components that share spatial overlap in their footprints.
    """

    component_axis: str = "components"
    """Name of the dimension representing individual components."""

    id_coordinates: str = "id_"
    """Name of the coordinate used to identify individual components with unique IDs."""

    type_coordinates: str = "type_"
    """Name of the coordinate used to specify component types (e.g., neuron, background)."""

    spatial_axes: tuple = ("height", "width")
    """Names of the dimensions representing spatial coordinates (height, width)."""

    def validate(self):
        """Validate parameter configurations.

        Raises:
            ValueError: If spatial_axes is not a tuple of length 2.
        """
        if not isinstance(self.spatial_axes, tuple) or len(self.spatial_axes) != 2:
            raise ValueError("spatial_axes must be a tuple of length 2")


@dataclass
class OverlapGroupsInitializer(SupervisedTransformer, metaclass=TransformerMeta):
    """Computes groups of spatially overlapping components.

    This transformer implements Algorithms 2 (DETERMINEGROUPS) and 3 (JOINGROUPS)
    to identify groups of components that share spatial overlap in their footprints.
    Components are grouped together if their spatial footprints have non-zero overlap.

    The algorithm processes components sequentially, either:
    - Adding a component to an existing group if it overlaps with any member
    - Creating a new group if the component doesn't overlap with existing groups

    The result is stored as a sparse matrix where non-zero elements indicate
    components belonging to the same overlap group.
    """

    params: OverlapGroupsInitializerParams
    """Configuration parameters for the overlap group computation."""

    overlap_groups_: xr.DataArray = field(init=False)
    """Computed sparse matrix indicating component group memberships."""

    def _join_groups(
        self,
        footprints: xr.DataArray,
        groups: List[Set[int]],
        component_idx: int,
        component: xr.DataArray,
    ) -> List[Set[int]]:
        """Implementation of Algorithm 3 (JOINGROUPS).

        Determines which group(s) a new component should join based on spatial overlap.

        Args:
            footprints: Spatial footprints of all components up to current index.
            groups: Current list of component groups.
            component_idx: Index of the new component.
            component: Spatial footprint of the new component.

        Returns:
            Updated list of component groups including the new component.
        """
        if not groups:
            return [{component_idx}]

        # Try to add to existing groups
        for group_idx, group in enumerate(groups):
            # Test for overlap with current group members
            has_overlap = False
            for member_idx in group:
                member = footprints[member_idx]
                if (component * member).sum() != 0:  # Test for spatial overlap
                    has_overlap = True
                    break

            if not has_overlap:
                continue

            # Add to existing group if overlap found
            groups[group_idx].add(component_idx)
            return groups

        # Create new group if no overlap with existing groups
        groups.append({component_idx})
        return groups

    def learn_one(self, footprints: Footprints, frame: xr.DataArray = None) -> Self:
        """Determine overlap groups from spatial footprints.

        This method implements Algorithm 2 (DETERMINEGROUPS) by processing
        components sequentially and grouping them based on spatial overlap.

        Args:
            frame: Not used
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)

        Returns:
            Self: The transformer instance for method chaining.
        """
        n_components = footprints.sizes[self.params.component_axis]
        groups: List[Set[int]] = []

        # Process components sequentially (Algorithm 2)
        for i in range(n_components):
            component = footprints[i]
            groups = self._join_groups(footprints[:i], groups, i, component)

        # Convert groups to sparse COO format
        coords = [], []  # (row, col) coordinates
        for group in groups:
            for i in group:
                for j in group:
                    coords[0].append(i)
                    coords[1].append(j)

        # Create sparse COO array
        data = np.ones(len(coords[0]), dtype=np.float32)
        sparse_matrix = sparse.COO(
            coords=coords, data=data, shape=(n_components, n_components)
        )

        # Create xarray DataArray with sparse data
        self.overlap_groups_ = xr.DataArray(
            sparse_matrix,
            dims=(self.params.component_axis, self.params.component_axis),
            coords={
                self.params.id_coordinates: (
                    self.params.component_axis,
                    footprints.coords[self.params.id_coordinates].values,
                ),
                self.params.type_coordinates: (
                    self.params.component_axis,
                    footprints.coords[self.params.type_coordinates].values,
                ),
            },
        )

        return self

    def transform_one(self, _=None) -> OverlapGroups:
        """Return the computed overlap groups.

        This method wraps the overlap groups matrix in an OverlapGroups object
        for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            OverlapGroups: Wrapped sparse matrix indicating component group memberships.
        """
        return OverlapGroups(self.overlap_groups_)
