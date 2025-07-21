import xarray as xr
import numpy as np


def is_non_negative(da: xr.DataArray) -> None:
    if da.min() < 0:
        raise ValueError("Array is not non-negative")


def is_unique(da: xr.DataArray) -> None:
    _, counts = np.unique(da, return_counts=True)
    if counts.max() > 1:
        raise ValueError("The values in DataArray are not unique.")


def is_unit_interval(da: xr.DataArray) -> None:
    if da.min() < 0 or da.max() > 1:
        raise ValueError("The values in DataArray are not unit interval.")
