from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from cala.models.axis import AXIS
from cala.streaming.util.new import package_frame


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
    assert dataarray.dims == AXIS.spatial_dims

    # Check the coordinates
    assert AXIS.frame_coord in dataarray.coords
    assert AXIS.timestamp_coord in dataarray.coords
    assert dataarray.coords[AXIS.frame_coord].item() == index
    assert pd.Timestamp(dataarray.coords[AXIS.timestamp_coord].values) == pd.Timestamp(timestamp)

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
    assert dataarray.dims == AXIS.spatial_dims

    # Check the coordinates
    assert AXIS.frame_coord in dataarray.coords
    assert AXIS.timestamp_coord in dataarray.coords
    assert dataarray.coords[AXIS.frame_coord].item() == index
    assert isinstance(dataarray.coords[AXIS.timestamp_coord].item(), str)

    # Check the data
    np.testing.assert_array_equal(dataarray.values, frame)

    # Check the name
    assert dataarray.name == "frame"
