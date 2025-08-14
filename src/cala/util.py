from datetime import datetime
from typing import Sequence
from uuid import uuid4

import numpy as np
import xarray as xr

from cala.models import AXIS


def package_frame(
    frame: np.ndarray, index: int, timestamp: datetime | str | None = None
) -> xr.DataArray:
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

    return frame.assign_coords(
        {
            AXIS.width_dim: range(frame.sizes[AXIS.width_dim]),
            AXIS.height_dim: range(frame.sizes[AXIS.height_dim]),
        }
    )


def create_id() -> str:
    return uuid4().hex


def combine_attr_replaces(attrs: Sequence[dict[str, list[str]]], context: None = None) -> dict:
    repl = [item for attr in attrs for item in attr.get("replaces", [])]
    return {"replaces": repl} if repl else {}
