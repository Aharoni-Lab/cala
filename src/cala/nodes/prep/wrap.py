from collections.abc import Generator
from datetime import datetime
from itertools import count
from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name

from cala.assets import Frame
from cala.models import AXIS


def counter(start: int = 0, limit: int = 1e7) -> A[Generator[int], Name("idx")]:
    cnt = count(start=start)
    while (val := next(cnt)) < limit:
        yield val


def package_frame(frame: np.ndarray, index: int, timestamp: datetime | str | None = None) -> Frame:
    """Transform a 2D numpy frame into an xarray DataArray.

    Args:
        frame: 2D numpy array representing the frame
        index: Index of the frame in the sequence
        timestamp: Timestamp of the frame capture - must be a string

    Returns:
        xr.DataArray: The frame packaged as a DataArray with axes and index
    """
    if timestamp is None:
        timestamp = datetime.now()  # means nothing. filler so that they're unique.

    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime("%H:%M:%S.%f")

    frame = xr.DataArray(
        frame,
        dims=AXIS.spatial_dims,
        coords={
            AXIS.frame_coord: index,
            AXIS.timestamp_coord: timestamp,
        },
        name="frame",
    )

    da = frame.assign_coords(
        {
            AXIS.width_dim: range(frame.sizes[AXIS.width_dim]),
            AXIS.height_dim: range(frame.sizes[AXIS.height_dim]),
        }
    )
    return Frame.from_array(da)
