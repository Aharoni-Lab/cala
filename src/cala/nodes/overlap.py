import numpy as np
import xarray as xr
from sparse import COO

from cala.assets import Footprints, Overlaps
from cala.models import AXIS
from cala.util import sp_matmul, stack_sparse, concatenate_coordinates


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
    no_new = new_footprints.array is None
    if no_new:
        return overlaps

    no_overlaps = overlaps.array is None
    if no_overlaps:
        return initialize(overlaps, new_footprints)

    A = stack_sparse(footprints.array, AXIS.component_dim).tocsr()
    V = overlaps.array.data.tocsr()

    a_new = new_footprints.array

    merged_ids = a_new.attrs.get("replaces", [])
    intact_mask = ~np.isin(overlaps.array[AXIS.id_coord].values, merged_ids)
    V_side = overlaps.array[AXIS.component_dim]

    if merged_ids:
        A = A[intact_mask]
        V = V[intact_mask].T[intact_mask]  # symmetric matrix
        V_side = V_side[intact_mask]

    a_sparse = stack_sparse(a_new, AXIS.component_dim).tocsr()

    # Compute spatial overlaps between new and existing components
    v_topright = (A @ a_sparse.T).nonzero()
    v_bottleft = v_topright[::-1]

    # Compute overlaps between new components themselves
    v_botright = a_sparse @ a_sparse.T

    # Construct the new overlap matrix by blocks
    # [     V      v_topright]
    # [v_bottleft  v_botright]
    updated_overlaps = assemble_sparse_bool(
        V.nonzero(), v_topright, v_bottleft, v_botright.nonzero(), V.shape, v_botright.shape
    )

    overlaps.array = overlap_format(updated_overlaps, V_side, a_new)

    return overlaps


def overlap_format(array: COO, V_comp: xr.DataArray, a_new_comp: xr.DataArray) -> xr.DataArray:
    prim_coords = concatenate_coordinates(V_comp.coords, a_new_comp.coords)
    seco_coords = concatenate_coordinates(
        V_comp.rename(AXIS.component_rename).coords, a_new_comp.rename(AXIS.component_rename).coords
    )
    return xr.DataArray(
        array,
        dims=(AXIS.component_dim, f"{AXIS.component_dim}'"),
        coords={k: (AXIS.component_dim, v) for k, v in prim_coords.items()},
    ).assign_coords({k: (f"{AXIS.component_dim}'", v) for k, v in seco_coords.items()})


def assemble_sparse_bool(
    top_left: tuple[np.ndarray, np.ndarray],
    top_right: tuple[np.ndarray, np.ndarray],
    bottom_left: tuple[np.ndarray, np.ndarray],
    bottom_right: tuple[np.ndarray, np.ndarray],
    init_shape: tuple[int, int],
    attach_shape: tuple[int, int],
) -> COO:
    """
    Assemble a sparse boolean array with four coordinates in the format of
    scipy.sparse.sp_matrix.nonzero()

    """
    x_coords = top_left[0]
    y_coords = top_left[1]
    x_coords = np.concatenate([x_coords, top_right[0]])
    y_coords = np.concatenate([y_coords, top_right[1] + init_shape[1]])
    x_coords = np.concatenate([x_coords, bottom_left[0] + init_shape[0]])
    y_coords = np.concatenate([y_coords, bottom_left[1]])
    x_coords = np.concatenate([x_coords, bottom_right[0] + init_shape[0]])
    y_coords = np.concatenate([y_coords, bottom_right[1] + init_shape[1]])

    final_shape = tuple(x1 + x2 for x1, x2 in zip(init_shape, attach_shape))
    return COO(coords=(x_coords, y_coords), shape=final_shape, data=1)


def assemble_square(
    top_left: np.ndarray, top_right: np.ndarray, bottom_left: np.ndarray, bottom_right: np.ndarray
) -> np.ndarray:
    """
    Assemble four 2D arrays into a single 2D array, with one in each corner.

    """
    top_block = np.hstack([top_left, top_right])
    bottom_block = np.hstack([bottom_left, bottom_right])

    # Finally combine top and bottom blocks
    return np.vstack([top_block, bottom_block])
