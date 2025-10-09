import numpy as np
import xarray as xr

from cala.assets import Footprints, Overlaps
from cala.models import AXIS
from cala.util import sp_matmul


def initialize(overlaps: Overlaps, footprints: Footprints) -> Overlaps:
    A = footprints.array

    if A is None:
        return overlaps

    V = sp_matmul(left=A, dim=AXIS.component_dim, rename_map=AXIS.component_rename)

    overlaps.array = V > 0

    return overlaps


def ingest_component(
    overlaps: Overlaps, footprints: Footprints, new_footprints: Footprints
) -> Overlaps:
    """Update component overlap matrix with new components.

    Updates the binary adjacency matrix that represents component overlaps.
    Matrix element (i,j) is 1 if components i and j overlap spatially, 0 otherwise.

    Args:
        footprints (Footprints): Current spatial footprints [A, b]
        new_footprints (Footprints): Newly detected spatial components
    """
    if new_footprints.array is None:
        return overlaps

    elif overlaps.array is None or overlaps.array.size == 1:
        return initialize(overlaps, footprints)

    V = overlaps.array

    a_new = new_footprints.array.transpose(AXIS.component_dim, ...)

    merged_ids = a_new.attrs.get("replaces", [])
    intact_ids = [id_ for id_ in V[AXIS.id_coord].values if id_ not in merged_ids]

    if merged_ids:
        V = (
            V.set_xindex(AXIS.id_coord)
            .set_xindex(f"{AXIS.id_coord}'")
            .sel({AXIS.id_coord: intact_ids, f"{AXIS.id_coord}'": intact_ids})
            .reset_index([AXIS.id_coord, f"{AXIS.id_coord}'"])
        )

    if a_new[AXIS.id_coord].item() in V[AXIS.id_coord].values:
        # trace REPLACEMENT
        dim_idx = np.where(V[AXIS.id_coord].values == a_new[AXIS.id_coord].item())[0].tolist()
        V = V.drop_sel({AXIS.component_dim: dim_idx, f"{AXIS.component_dim}'": dim_idx})

    # Also have to remove the ID from A,
    # since it's been already added in footprints.component_ingest
    A = footprints.array
    id_idx = np.where(A[AXIS.id_coord].values == a_new[AXIS.id_coord].item())[0].tolist()
    A = A.drop_sel({AXIS.component_dim: id_idx})

    # Compute spatial overlaps between new and existing components
    bl_overlap = sp_matmul(
        left=A, right=a_new, dim=AXIS.component_dim, rename_map=AXIS.component_rename
    )
    tr_overlap = xr.DataArray(
        bl_overlap.data,
        dims=[f"{AXIS.component_dim}'", AXIS.component_dim],
        coords=a_new[AXIS.component_dim].coords,
    ).assign_coords(A[AXIS.component_dim].rename(AXIS.component_rename).coords)

    # Compute overlaps between new components themselves
    new_overlaps = sp_matmul(left=a_new, dim=AXIS.component_dim, rename_map=AXIS.component_rename)

    # Construct the new overlap matrix by blocks
    # [existing_overlaps    og_new_overlaps.T]
    # [og_new_overlaps        new_overlaps   ]

    # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
    top_block = xr.concat([V.astype(float), tr_overlap], dim=AXIS.component_dim)

    # Then concatenate vertically with [new_overlaps, new_overlaps]
    bottom_block = xr.concat([bl_overlap, new_overlaps], dim=AXIS.component_dim)

    # Finally combine top and bottom blocks
    updated_overlaps = xr.concat([top_block, bottom_block], dim=f"{AXIS.component_dim}'")

    overlaps.array = updated_overlaps > 0

    return overlaps
