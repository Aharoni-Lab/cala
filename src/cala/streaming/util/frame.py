from datetime import datetime

import numpy as np
import xarray as xr

from cala.streaming.core.axis import Axis


def package_frame(frame: np.ndarray, index: int, timestamp: datetime | None = None) -> xr.DataArray:
    """Transform a 2D numpy frame into an xarray DataArray.

    Args:
        frame: 2D numpy array representing the frame
        index: Index of the frame in the sequence
        timestamp: Timestamp of the frame capture

    Returns:
        xr.DataArray: The frame packaged as a DataArray with axes and index
    """
    return xr.DataArray(
        frame,
        dims=Axis.spatial_axes,
        coords={
            Axis.frame_coordinates: index,
            Axis.time_coordinates: timestamp,
        },
        name="frame",
    )
