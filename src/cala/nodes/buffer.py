from typing import Annotated as A

from noob import Name

from cala.assets import Frame, Buffer
from cala.models import AXIS


def fill_buffer(buffer: Buffer, frame: Frame) -> A[Buffer, Name("buffer")]:
    if buffer.array is None:
        buffer.array = frame.array.volumize.dim_with_coords(
            dim=AXIS.frames_dim, coords=[AXIS.timestamp_coord]
        )
        return buffer

    buffer.append(frame.array)
    return buffer
