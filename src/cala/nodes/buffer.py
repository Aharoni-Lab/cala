from typing import Annotated as A

import xarray as xr
from noob import Name

from cala.assets import Frame, Movie
from cala.models import AXIS


def fill_buffer(size: int, buffer: Movie, frame: Frame) -> A[Movie, Name("buffer")]:
    if buffer.array is None:
        buffer.array = frame.array.volumize.dim_with_coords(
            dim=AXIS.frames_dim, coords=[AXIS.timestamp_coord]
        )
        return buffer

    buffered = (
        buffer.array.transpose(AXIS.frames_dim, ...)[-size + 1 :]
        if buffer.array.sizes[AXIS.frames_dim] >= size
        else buffer.array
    )

    buffer.array = xr.concat([buffered, frame.array], dim=AXIS.frames_dim)
    return buffer
