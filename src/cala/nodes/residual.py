from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from sparse import COO

from cala.assets import Buffer, Footprints, Frame, Traces
from cala.models import AXIS


def build(
    residuals: Buffer,
    frame: Frame,
    footprints: Footprints,
    traces: Traces,
    n_recalc: int,
) -> A[Buffer, Name("movie")]:
    """
    Computes and maintains a buffer of residual signals.

    This method implements the residual computation by subtracting the
    reconstructed signal from the original data. It maintains only the
    most recent frames as specified by the buffer length.

    The residual buffer contains the recent history of unexplained variance
    in the data after accounting for known components.

    The computation follows the equation:
        R_buf = [Y − [A, b][C; f]][:, t′ − l_b + 1 : t′]
        where:
        - Y is the data matrix (pixels × time)
        - [A, b] is the spatial footprint matrix of neurons and background
        - [C; f] is the temporal traces matrix of neurons and background
        - t' is the current timestep
        - l_b is the buffer length
        - R_buf is the resulting residual buffer

    Args:
        footprints (Footprints): Spatial footprints of all components.
            Shape: (components × height × width)
        traces (Traces): Temporal traces of all components.
            Shape: (components × time)
        frames (Movie): Stack of frames up to current timestep.
            Shape: (frames × height × width)
    """
    if footprints.array is None or traces.array is None:
        if residuals.array is None:
            residuals.array = frame.array.expand_dims(dim=AXIS.frames_dim)
        else:
            residuals.append(frame.array)
        return residuals

    Y = frame.array
    C = traces.array.isel({AXIS.frames_dim: -1})  # (components,)
    A = footprints.array

    R_curr, flag = _find_overestimates(Y=Y, A=A, C=C)
    if flag:
        C = _align_overestimates(A, C, R_curr)
        traces.array.loc[{AXIS.frames_dim: -1}] = C

    # if recently discovered, set to zero (or a small number). otherwise, just append
    preserve_area = _get_new_estimators_area(A=A, C=C, n_recalc=n_recalc)
    if preserve_area is not None:
        residuals.array_ *= preserve_area.as_numpy()
    R_curr = _get_residuals(Y=Y, A=A, C=C)
    # we're not fully modifying for the negative minimum, so we need to clip
    residuals.append(R_curr.clip(min=0))

    return residuals


def _get_residuals(Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray) -> xr.DataArray:
    return Y - xr.DataArray(
        np.matmul(A.transpose(*AXIS.spatial_dims, ...).data, C.data), dims=AXIS.spatial_dims
    )


def _find_overestimates(
    Y: xr.DataArray, A: xr.DataArray, C: xr.DataArray
) -> tuple[xr.DataArray, bool]:
    R_curr = _get_residuals(Y, A, C)
    return R_curr, R_curr.min() < -np.finfo(np.float32).eps


def _align_overestimates(
    A: xr.DataArray, C_latest: xr.DataArray, R_latest: xr.DataArray
) -> xr.DataArray:
    """
    !!We're assuming there's no completely occluded component. This might be a problem eventually!!
    """
    A_pix = A.data.reshape((A.sizes[AXIS.component_dim], -1)).tocsr()
    R = R_latest.values
    unlayered_stamp = _find_unlayered_footprints(A_pix)  # same up to here

    R_rel = unlayered_stamp * np.minimum(R, 0).reshape(1, -1)  # same up to here to 2e-6 and nan
    RA = A_pix.power(-1).multiply(R_rel).tocsr()  # same up to here to 2e-6 and neginf
    dC = RA.minimum(0).sum(axis=1)  # .nanmin(axis=1, explicit=True)
    # divide by the number of active pixels to normalize negative (prevents outliers)
    dC_norm = np.asarray(dC).squeeze() / np.array([a.nnz for a in RA])

    return (C_latest + dC_norm).clip(min=0)


def _find_unlayered_footprints(A: COO) -> np.ndarray:
    coords = A.nonzero()[1]
    pixels, counts = np.unique(coords, return_counts=True)
    mask = np.isin(coords, pixels[counts == 1])
    vals = A.data[mask]
    locs = coords[mask]
    ret = np.zeros(A.shape[1])
    ret[locs] = vals
    return ret
    # return coo_matrix((vals, (np.zeros_like(mask), mask)), shape=(1, A.shape[1])).toarray()


def _get_new_estimators_area(
    A: xr.DataArray, C: xr.DataArray, n_recalc: int
) -> xr.DataArray | None:
    targets = C[AXIS.detect_coord].values >= (C[AXIS.frame_coord].item() - n_recalc)

    if any(targets):
        idx = np.where(targets)[0]
        nonzeros = A.data.nonzero()
        target_mask = np.isin(nonzeros[0], idx)
        target_coords = tuple(nonzero[target_mask] for nonzero in nonzeros[1:])
        target_area = np.ones(A.shape[1:], dtype=bool)
        target_area[target_coords] = 0
        return xr.DataArray(target_area, dims=A.dims[1:])
    else:
        return None
