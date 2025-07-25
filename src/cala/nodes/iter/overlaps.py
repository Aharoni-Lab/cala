import sparse
import xarray as xr
from noob.node import Node

from cala.models import AXIS, Footprints, Overlap


class Overlaps(Node):
    overlaps_: Overlap = None

    def process(self, footprints: Footprints, new_footprints: Footprints = None):
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
    ) -> Overlap:
        """
        Sparse matrix of component footprint overlaps.

        Args:
            footprints (Footprints): Current temporal component c_t.
        """
        A = footprints.array

        # Use matrix multiplication with broadcasting to compute overlaps
        data = (
            A.dot(
                A.rename(
                    {
                        AXIS.component_dim: f"{AXIS.component_dim}'",
                        AXIS.id_coord: f"{AXIS.id_coord}'",
                        AXIS.confidence_coord: f"{AXIS.confidence_coord}'",
                    }
                )
            )
            > 0
        ).astype(int)

        # Create xarray DataArray with sparse data
        data.values = sparse.COO(data.values)
        self.overlaps_ = Overlap(
            array=data.assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, A[AXIS.id_coord].values),
                    AXIS.confidence_coord: (AXIS.component_dim, A[AXIS.confidence_coord].values),
                }
            )
        )

        return self.overlaps_

    def ingest_frame(self, footprints: Footprints) -> Overlap:
        return self.initialize(footprints)

    def ingest_component(
        self,
        footprints: Footprints,
        new_footprints: Footprints,
    ) -> Overlap:
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
        old_new_overlap = A.dot(
            a_new.rename({self.params.component_dim: f"{self.params.component_dim}'"})
        )
        bottom_left_overlap = (
            (old_new_overlap != 0)
            .astype(int)
            .assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, A[AXIS.id_coord].values),
                    AXIS.type_coord: (AXIS.component_dim, A[AXIS.type_coord].values),
                }
            )
        )

        bottom_left_overlap.values = sparse.COO(bottom_left_overlap.values)

        top_right_overlap = xr.DataArray(
            bottom_left_overlap,
            dims=bottom_left_overlap.dims[::-1],
            coords=a_new.coords,
        )

        # Compute overlaps between new components themselves
        new_new_overlaps = a_new.dot(a_new.rename({AXIS.component_dim: f"{AXIS.component_dim}'"}))
        new_new_overlaps = (new_new_overlaps != 0).astype(int).assign_coords(a_new.coords)

        new_new_overlaps.values = sparse.COO(new_new_overlaps.values)

        # Construct the new overlap matrix by blocks
        # [existing_overlaps    new_overlaps.T    ]
        # [new_overlaps        new_new_overlaps   ]

        # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
        top_block = xr.concat([self.overlaps_, top_right_overlap], dim=AXIS.component_dim)

        # Then concatenate vertically with [new_overlaps, new_new_overlaps]
        bottom_block = xr.concat([bottom_left_overlap, new_new_overlaps], dim=AXIS.component_dim)

        # Finally combine top and bottom blocks
        updated_overlaps = xr.concat([top_block, bottom_block], dim=f"{AXIS.component_dim}'")

        return Overlap(array=updated_overlaps)
