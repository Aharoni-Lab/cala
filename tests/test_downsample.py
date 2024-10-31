import numpy as np
import xarray as xr
import pytest

from cala.preprocess.downsample import Downsampler


def test_downsampler_mean():
    # Create a sample DataArray with arbitrary dimensions
    data = np.random.rand(100, 200, 300)  # Dimensions: (time, x, y)
    dims = ("time", "x", "y")
    coords = {
        "time": np.arange(100),
        "x": np.arange(200),
        "y": np.arange(300),
    }
    X = xr.DataArray(data, coords=coords, dims=dims, name="sample_data")

    # Chunk the DataArray for Dask (optional)
    X = X.chunk({"time": 10, "x": 50, "y": 50})

    # Instantiate the Downsampler
    downsampler = Downsampler(method="mean", dims=("time", "x", "y"), strides=(2, 4, 5))

    # Apply the transform
    downsampled_X = downsampler.transform(X)

    # Compute the result if using Dask
    downsampled_X = downsampled_X.compute()

    # Expected shape after downsampling
    expected_shape = (
        X.sizes["time"] // 2,
        X.sizes["x"] // 4,
        X.sizes["y"] // 5,
    )

    # Assertions to verify the downsampling
    assert (
        downsampled_X.shape == expected_shape
    ), "Downsampled shape does not match expected shape."
    assert (
        downsampled_X.dims == X.dims
    ), "Dimensions of downsampled DataArray do not match original."
    assert (
        downsampled_X.name == X.name
    ), "Name of downsampled DataArray does not match original."
    assert downsampled_X.notnull().all(), "Downsampled DataArray contains null values."

    print("Original shape:", X.shape)
    print("Downsampled shape:", downsampled_X.shape)
