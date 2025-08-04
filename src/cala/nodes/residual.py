import xarray as xr

from cala.assets import Footprints, Frame, Movie, PopSnap, Residual, Traces
from cala.models import AXIS


def initialize(frames: Movie, footprints: Footprints, traces: Traces) -> Residual:
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
    # Reshape frames to pixels x time
    Y = frames.array

    # Get temporal components [C; f]
    C = traces.array  # components x time

    # Reshape footprints to (pixels x components)
    A = footprints.array

    # Compute residual R = Y - [A,b][C;f]
    R = Y - (A @ C)

    return Residual.from_array(R)


def ingest_frame(
    residual: Residual, frame: Frame, footprints: Footprints, traces: PopSnap
) -> Residual:
    """
    Update residual buffer with new frame, footprints, and traces
    """
    new_R = frame.array - footprints.array @ traces.array

    residual.array = xr.concat([residual.array, new_R], dim=AXIS.frames_dim)

    return residual
