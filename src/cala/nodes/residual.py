from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name

from cala.assets import Footprints, Movie, Residual, Traces
from cala.models import AXIS


def build(
    residuals: Residual, frames: Movie, footprints: Footprints, traces: Traces
) -> A[Residual, Name("movie")]:
    """
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
        return Residual.from_array(frames.array)

    # Reshape frames to pixels x time
    Y = frames.array

    # Get temporal components [C; f]
    C = traces.array.sel(
        {AXIS.frame_coord: Y[AXIS.frame_coord].values.tolist()}
    )  # components x time

    # Reshape footprints to (pixels x components)
    A = footprints.array

    R_latest = Y.isel({AXIS.frames_dim: -1}) - (A @ C.isel({AXIS.frames_dim: -1}))
    if R_latest.min() < 0:
        shifted_tr = _align_overestimates(A, C.isel({AXIS.frames_dim: -1}), R_latest)
        C.loc[{AXIS.frames_dim: C[AXIS.frame_coord].max()}] = shifted_tr
        traces.array.loc[{AXIS.frames_dim: C[AXIS.frame_coord].max()}] = shifted_tr

    # Compute residual R = Y - [A,b][C;f]
    R = Y - (A @ C)
    residuals.array = R.clip(min=0)  # clipping is for the first n frames

    return residuals


def _align_overestimates(
    A: xr.DataArray, C_latest: xr.DataArray, R_latest: xr.DataArray
) -> xr.DataArray:
    """
        Gotta be able to do at least ONE OF splitoff or gradualon.

        Negative residuals just need to go. There isn't much you can do with the value...?

        Two cases: (A & B Overlapping)
        1. GradualOn: Know A. B turns ON
            -> trace tries to chase (increases)
            -> footprint tries to chase
            -> residual becomes negative at A-B
                -> should just decrease, positive at A^B
                -> actually... should decrease (just more steeply)

        2. SplitOff: Know AB. B turns OFF
            -> trace tries to chase (decreases)
            -> footprint tries to chase
            -> residual becomes positive at A-B
                -> should increase, negative at A^B
                -> this should just decrease, MORE negative at B-A
                -> this going to zero makes sense
            OR
            keep B, remove A-B

            R = Y - A @ C

        What about the past frame residuals after?

        for GradualOn, nothing should go to zero.
        for SplitOff, a chunk needs to go to zero.

            So... how about we do something like (if it's been on for a long time,
            we become less likely to purge it?)

    We subsequently clip R minimum to zero, since all significant negative residual spots
    have been removed, and the remaining negative spots are noise level.

    !!We're assuming there's no completely occluded component. This might be a problem eventually!!
    """

    unlayered_footprints = _find_unlayered_footprints(A)
    # if unlayered_footprints.max(dim=AXIS.spatial_dims).min() == 0:
    #     raise ValueError("There are at least one completely occluded components.")

    R_rel = R_latest.where((R_latest < 0) * unlayered_footprints.max(dim=AXIS.component_dim))
    dC = (
        (R_rel / A)
        .min(dim=AXIS.spatial_dims)
        .reset_coords([AXIS.frame_coord, AXIS.timestamp_coord], drop=True)
    )

    return (C_latest + xr.apply_ufunc(np.nan_to_num, dC, kwargs={"neginf": 0})).clip(min=0)


def _find_unlayered_footprints(A: xr.DataArray) -> xr.DataArray:
    A_layer_mask = (A > 0).sum(dim=AXIS.component_dim)
    return A.where(A_layer_mask == 1, 0)
