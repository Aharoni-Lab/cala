from typing import Annotated as A

import xarray as xr
from noob import Name

from cala.assets import Frame, Movie
from cala.models import AXIS


def fill_buffer(size: int, buffer: Movie, frame: Frame) -> A[Movie, Name("buffer")]:
    if buffer.array is None:
        buffer.array = frame.array.expand_dims(AXIS.frames_dim).assign_coords(
            {AXIS.timestamp_coord: (AXIS.frames_dim, [frame.array[AXIS.timestamp_coord].item()])}
        )
        return buffer

    buffered = (
        buffer.array.transpose(AXIS.frames_dim, ...)[-size + 1 :]
        if buffer.array.sizes[AXIS.frames_dim] >= size
        else buffer.array
    )

    buffer.array = xr.concat([buffered, frame.array], dim=AXIS.frames_dim)
    return buffer
