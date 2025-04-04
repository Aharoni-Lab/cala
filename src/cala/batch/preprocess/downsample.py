from dataclasses import dataclass, field
from typing import Literal, Self

import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Downsampler(BaseEstimator, TransformerMixin):
    """
    A transformer that downsamples an xarray DataArray along specified dimensions using either
    'mean' or 'subset' methods.
    """

    # method (str): The downsampling method to use ('mean' or 'subset').
    method: Literal["mean", "subset"] = "mean"
    # dimensions (tuple of str): The dimensions along which to downsample.
    dimensions: list[str] = field(default_factory=lambda: ["frames", "width", "height"])
    # strides (tuple of int): The strides or pool sizes for each dimension.
    strides: list[int] = field(default_factory=lambda: [1, 1, 1])
    # keyword arguments for each downsampling method
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in ("mean", "subset"):
            raise ValueError(f"Downsampling method '{self.method}' not understood.")
        if len(self.dimensions) != len(self.strides):
            raise ValueError("Length of 'dims' and 'strides' must be equal.")

    def fit(self, X: xr.DataArray, y: None = None) -> Self:
        return self

    def transform(self, X: xr.DataArray, y: None = None) -> xr.DataArray:
        """
        Downsample the DataArray X.

        Parameters:
            X (xr.DataArray): The input DataArray to downsample.

        Returns:
            xr.DataArray: The downsampled DataArray.
        """
        if self.method == "mean":
            return self.mean_downsample(X)
        elif self.method == "subset":
            return self.subset_downsample(X)

    def mean_downsample(self, array: xr.DataArray) -> xr.DataArray:
        """
        Downsample the array by taking the mean over specified window sizes.

        Parameters:
            array (xr.DataArray): The DataArray to downsample.

        Returns:
            xr.DataArray: The downsampled DataArray.
        """
        coarsen_dims = {dim: stride for dim, stride in zip(self.dimensions, self.strides)}
        return array.coarsen(coarsen_dims, boundary="trim").mean(**self.kwargs)

    def subset_downsample(self, array: xr.DataArray) -> xr.DataArray:
        """
        Downsample the array by subsetting (taking every nth element) over specified dimensions.

        Parameters:
            array (xr.DataArray): The DataArray to downsample.

        Returns:
            xr.DataArray: The downsampled DataArray.
        """
        indexers = {
            dim: slice(None, None, stride) for dim, stride in zip(self.dimensions, self.strides)
        }
        return array.isel(indexers)
