from typing import Annotated as A
from noob import Name
import xarray as xr
from cala.models import Movie, Frame, AXIS


def fill_buffer(size: int, buffer: Movie, frame: Frame) -> A[Movie, Name("buffer")]:
    if buffer.array is None:
        buffer.array = frame.array.expand_dims(AXIS.frames_dim).assign_coords(
            {AXIS.timestamp_coord: (AXIS.frames_dim, [frame.array[AXIS.timestamp_coord].item()])}
        )
        return buffer

    buffered = (
        buffer.array.transpose(AXIS.frames_dim, ...)[-size:]
        if buffer.array.sizes[AXIS.frames_dim] >= size
        else buffer.array
    )

    buffer.array = xr.concat([buffered, frame], dim=AXIS.frames_dim)
    return buffer
