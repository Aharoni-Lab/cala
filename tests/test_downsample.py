import numpy as np
import xarray as xr
import pytest

from cala.preprocess.downsample import Downsampler


@pytest.fixture
def example_xarray():
    data = np.random.rand(100, 200, 300)  # Dimensions: (time, x, y)
    dims = ("time", "x", "y")
    coords = {
        "time": np.arange(100),
        "x": np.arange(200),
        "y": np.arange(300),
    }
    X = xr.DataArray(data, coords=coords, dims=dims, name="sample_data")
    return X


def test_downsampler_mean(example_xarray):
    X = example_xarray.chunk({"time": 10, "x": 50, "y": 50})

    downsampler = Downsampler(method="mean", dims=("time", "x", "y"), strides=(2, 4, 5))
    downsampled_X = downsampler.transform(X)
    downsampled_X = downsampled_X.compute()

    expected_shape = (
        X.sizes["time"] // 2,
        X.sizes["x"] // 4,
        X.sizes["y"] // 5,
    )

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


def test_downsampler_subset(example_xarray):
    # Instantiate the Downsampler with 'subset' method
    downsampler = Downsampler(
        method="subset", dims=("time", "x", "y"), strides=(2, 4, 5)
    )

    # Apply the transform
    downsampled_X = downsampler.transform(example_xarray)

    # Expected shape
    expected_shape = (
        example_xarray.sizes["time"] // 2
        + (1 if example_xarray.sizes["time"] % 2 else 0),
        example_xarray.sizes["x"] // 4 + (1 if example_xarray.sizes["x"] % 4 else 0),
        example_xarray.sizes["y"] // 5 + (1 if example_xarray.sizes["y"] % 5 else 0),
    )

    # Assertions
    assert (
        downsampled_X.shape == expected_shape
    ), "Downsampled shape does not match expected shape."


@pytest.mark.parametrize(
    "method,strides",
    [
        ("mean", (2, 4, 5)),
        ("mean", (5, 10, 15)),
        ("subset", (3, 6, 9)),
        ("subset", (4, 8, 12)),
    ],
)
def test_downsampler_methods(method, strides, example_xarray):
    X = example_xarray.chunk({"time": 10, "x": 50, "y": 50})
    downsampler = Downsampler(method=method, dims=example_xarray.dims, strides=strides)
    downsampled_X = downsampler.transform(X).compute()

    if method == "mean":
        expected_shape = tuple(
            X.sizes[dim] // stride for dim, stride in zip(example_xarray.dims, strides)
        )
    elif method == "subset":
        expected_shape = tuple(
            len(range(0, X.sizes[dim], stride))
            for dim, stride in zip(example_xarray.dims, strides)
        )
    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    assert (
        downsampled_X.shape == expected_shape
    ), f"{method} downsampling failed for strides {strides}"
