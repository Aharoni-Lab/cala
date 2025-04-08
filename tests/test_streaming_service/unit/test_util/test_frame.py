from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from cala.streaming.core.axis import Axis
from cala.streaming.util.frame import package_frame


def test_package_frame():
    # Create a sample 2D numpy array
    frame = np.random.randint(0, 256, size=(100, 200), dtype=np.uint16)
    index = 5
    timestamp = datetime(2023, 4, 8, 12, 0, 0)

    # Transform the frame
    dataarray = package_frame(frame, index, timestamp)

    # Check the type
    assert isinstance(dataarray, xr.DataArray)

    # Check the dimensions
    assert dataarray.dims == Axis.spatial_axes

    # Check the coordinates
    assert Axis.frame_idx_coordinates in dataarray.coords
    assert Axis.timestamp_coordinates in dataarray.coords
    assert dataarray.coords[Axis.spatial_axes[0]].size == frame.shape[0]
    assert dataarray.coords[Axis.spatial_axes[1]].size == frame.shape[1]
    assert dataarray.coords[Axis.frame_idx_coordinates].item() == index
    assert pd.Timestamp(dataarray.coords[Axis.timestamp_coordinates].values) == pd.Timestamp(
        timestamp
    )

    # Check the data
    np.testing.assert_array_equal(dataarray.values, frame)

    # Check the name
    assert dataarray.name == "frame"


def test_package_frame_datetimeless():
    # Create a sample 2D numpy array
    frame = np.random.randint(0, 256, size=(100, 200), dtype=np.uint16)
    index = 5

    # Transform the frame
    dataarray = package_frame(frame, index)

    # Check the type
    assert isinstance(dataarray, xr.DataArray)

    # Check the dimensions
    assert dataarray.dims == Axis.spatial_axes

    # Check the coordinates
    assert Axis.frame_idx_coordinates in dataarray.coords
    assert Axis.timestamp_coordinates in dataarray.coords
    assert dataarray.coords[Axis.spatial_axes[0]].size == frame.shape[0]
    assert dataarray.coords[Axis.spatial_axes[1]].size == frame.shape[1]
    assert dataarray.coords[Axis.frame_idx_coordinates].item() == index
    assert not dataarray.coords[Axis.timestamp_coordinates].item()

    # Check the data
    np.testing.assert_array_equal(dataarray.values, frame)

    # Check the name
    assert dataarray.name == "frame"
