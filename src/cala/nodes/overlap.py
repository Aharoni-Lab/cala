import numpy as np
import xarray as xr

from cala.assets import Footprints, Overlaps
from cala.models import AXIS



def ingest_frame(overlaps: Overlaps, footprints: Footprints) -> Overlaps:
    A = footprints.array

    if A is None:
        return overlaps

    V = (A @ A.rename(AXIS.component_rename)) > 0

    overlaps.array = V

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
        overlaps.array = ingest_frame(overlaps, footprints).array
        return overlaps

    V = overlaps.array

    a_new = new_footprints.array.volumize.dim_with_coords(
        dim=AXIS.component_dim, coords=[AXIS.id_coord, AXIS.confidence_coord]
    )

    if a_new[AXIS.id_coord].item() in V[AXIS.id_coord].values:
        # trace REPLACEMENT
        dim_idx = np.where(V[AXIS.id_coord].values == a_new[AXIS.id_coord].item())[0].tolist()
        V = V.drop_sel({AXIS.component_dim: dim_idx, f"{AXIS.component_dim}'": dim_idx})

    # think i also have to remove the ID from A,
    # since it's been already added in footprints.component_ingest
    A = footprints.array
    id_idx = np.where(A[AXIS.id_coord].values == a_new[AXIS.id_coord].item())[0].tolist()
    A = A.drop_sel({AXIS.component_dim: id_idx})

    # Compute spatial overlaps between new and existing components
    bottom_left_overlap = A @ a_new.rename(AXIS.component_rename)
    top_right_overlap = A.rename(AXIS.component_rename) @ a_new

    # Compute overlaps between new components themselves
    new_overlaps = a_new @ a_new.rename(AXIS.component_rename)

    # Construct the new overlap matrix by blocks
    # [existing_overlaps    og_new_overlaps.T]
    # [og_new_overlaps        new_overlaps   ]

    # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
    top_block = xr.concat([V.astype(float), top_right_overlap], dim=AXIS.component_dim)

    # Then concatenate vertically with [new_overlaps, new_overlaps]
    bottom_block = xr.concat([bottom_left_overlap, new_overlaps], dim=AXIS.component_dim)

    # Finally combine top and bottom blocks
    updated_overlaps = xr.concat([top_block, bottom_block], dim=f"{AXIS.component_dim}'")

    overlaps.array = updated_overlaps > 0

    return overlaps
