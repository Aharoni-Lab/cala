from dataclasses import dataclass, field
from typing import Self

import sparse
import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints
from cala.streaming.stores.odl import Overlaps


@dataclass
class OverlapsInitializerParams(Parameters, Axis):
    """Parameters for computing spatially overlapping component groups.

    This class defines the configuration parameters needed for determining
    groups of components that share spatial overlap in their footprints.
    """

    def validate(self) -> None:
        """Validate parameter configurations."""
        pass


@dataclass
class OverlapsInitializer(SupervisedTransformer):
    """Computes groups of spatially overlapping components.

    This transformer identifies groups of components that share spatial overlap in their footprints.
    Components are grouped together if their spatial footprints have non-zero overlap.

    The result is stored as a sparse matrix where non-zero elements indicate
    components belonging to the same overlap group.
    """

    params: OverlapsInitializerParams
    """Configuration parameters for the overlap group computation."""

    overlaps_: xr.DataArray = field(init=False)
    """Computed sparse matrix indicating component group memberships."""

    def learn_one(self, footprints: Footprints, frame: xr.DataArray | None = None) -> Self:
        """Determine overlaps from spatial footprints.

        This method links components based on spatial overlap.

        Args:
            frame: Not used
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Use matrix multiplication with broadcasting to compute overlaps
        data = (
            footprints.dot(
                footprints.rename({self.params.component_axis: f"{self.params.component_axis}'"})
            )
            > 0
        ).astype(int)

        # Create xarray DataArray with sparse data
        data.values = sparse.COO(data.values)
        self.overlaps_ = data.assign_coords(footprints.coords)

        return self

    def transform_one(self, _: None = None) -> Overlaps:
        """Return the computed overlaps.

        This method wraps the overlaps matrix in an OverlapGroups object
        for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            Overlaps: Wrapped sparse matrix indicating component group memberships.
        """
        return self.overlaps_
