from typing import Annotated as A

import xarray as xr
from noob import Name
from skimage.restoration import estimate_sigma

from cala.assets import Footprints, Movie, Residual, Traces
from cala.models import AXIS


def build(
    frames: Movie,
    footprints: Footprints,
    traces: Traces,
    trigger: bool,
    clip_threshold: float | None = None,
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

    # Compute residual R = Y - [A,b][C;f]
    R = Y - (A @ C)

    clip_val = _estimate_clip_val(Y, clip_threshold)
    footprints.array = _clear_overestimates(A, R, clip_val)

    return Residual.from_array(R.clip(min=0))


def _estimate_clip_val(Y: xr.DataArray, clip_threshold: float | None = None) -> float:
    """
    Estimate the threshold of "what is a significant negative residual value?" (above noise level)

    :param Y:
    :param clip_threshold:
    :return:
    """
    if clip_threshold:
        return -Y.max().item() * clip_threshold
    else:
        return -estimate_sigma(Y)


def _clear_overestimates(A: xr.DataArray, R: xr.DataArray, clip_val: float) -> xr.DataArray:
    """
    Remove all sections of the footprints that cause negative residuals.
    This occurs by:
    1. find "significant" negative residual spots that is more than a noise level, and thus
    cannot be clipped to zero. !!!! (only of the latest frame, and then go back to trace update..?)
    2. all footprint values at these spots go to zero.


    We subsequently clip R minimum to zero, since all significant negative residual spots
    have been removed, and the remaining negative spots are noise level.
    """

    R_min = R.min(dim=AXIS.frames_dim)
    footprints = A.where(R_min > clip_val, 0, drop=False)

    return footprints
