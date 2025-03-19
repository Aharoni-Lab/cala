from dataclasses import dataclass, field
from typing import Self

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
        # Use matrix multiplication with broadcasting to compute overlaps
        data = (np.tensordot(footprints, footprints, axes=((1, 2), (1, 2))) > 0).astype(
            int
        )

        sparse_matrix = sparse.COO(data)

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
