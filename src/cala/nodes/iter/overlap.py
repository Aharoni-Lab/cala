import xarray as xr
from noob.node import Node

from cala.assets import Footprints, Overlaps
from cala.models import AXIS


class Overlapper(Node):
    overlaps_: Overlaps = None

    def process(self, footprints: Footprints, new_footprints: Footprints = None) -> Overlaps:
        """
        A jenky ass temporary process method to circumvent Resource not yet being implemented.
        """
        if new_footprints is None:
            return self.initialize(footprints)
        else:
            return self.ingest_component(footprints=footprints, new_footprints=new_footprints)

    def initialize(
        self,
        footprints: Footprints,
    ) -> Overlaps:
        """
        Sparse matrix of component footprint overlaps.

        Args:
            footprints (Footprints): Current temporal component c_t.
        """
        A = footprints.array

        # Use matrix multiplication with broadcasting to compute overlaps
        data = (A @ A.rename(AXIS.component_rename)) > 0

        # Create xarray DataArray with sparse data
        self.overlaps_ = Overlaps(array=data)

        return self.overlaps_

    def ingest_frame(self, footprints: Footprints) -> Overlaps:
        return self.initialize(footprints)

    def ingest_component(self, footprints: Footprints, new_footprints: Footprints) -> Overlaps:
        """Update component overlap matrix with new components.

        Updates the binary adjacency matrix that represents component overlaps.
        Matrix element (i,j) is 1 if components i and j overlap spatially, 0 otherwise.

        Args:
            footprints (Footprints): Current spatial footprints [A, b]
            new_footprints (Footprints): Newly detected spatial components
        """

        A = footprints.array
        a_new = new_footprints.array

        # Compute spatial overlaps between new and existing components
        bottom_left_overlap = A @ a_new.rename(AXIS.component_rename)
        top_right_overlap = A.rename(AXIS.component_rename) @ a_new

        # Compute overlaps between new components themselves
        new_overlaps = a_new @ a_new.rename(AXIS.component_rename)

        # Construct the new overlap matrix by blocks
        # [existing_overlaps    og_new_overlaps.T]
        # [og_new_overlaps        new_overlaps   ]

        # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
        top_block = xr.concat(
            [self.overlaps_.array.astype(float), top_right_overlap], dim=AXIS.component_dim
        )

        # Then concatenate vertically with [new_overlaps, new_overlaps]
        bottom_block = xr.concat([bottom_left_overlap, new_overlaps], dim=AXIS.component_dim)

        # Finally combine top and bottom blocks
        updated_overlaps = xr.concat([top_block, bottom_block], dim=f"{AXIS.component_dim}'")

        return Overlaps(array=updated_overlaps > 0)
