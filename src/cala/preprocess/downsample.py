from typing import Tuple, Literal, Hashable

import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


class Downsampler(BaseEstimator, TransformerMixin):
    """
    A transformer that downsamples an xarray DataArray along specified dimensions using either
    'mean' or 'subset' methods.
    """

    def __init__(
        self,
        method: Literal["mean", "subset"] = "mean",
        dims: Tuple[str | Hashable, ...] = ("time", "x", "y"),
        strides: Tuple[int, ...] = (1, 1, 1),
        **kwargs,
    ):
        """
        Initialize the Downsampler.

        Parameters:
            method (str): The downsampling method to use ('mean' or 'subset').
            dims (tuple of str): The dimensions along which to downsample.
            strides (tuple of int): The strides or pool sizes for each dimension.
            **kwargs: Additional keyword arguments for the downsampling methods.
        """
        if method not in ("mean", "subset"):
            raise ValueError(
                f"Downsampling method '{method}' not understood. "
                f"Available methods are: 'mean', 'subset'"
            )
        if len(dims) != len(strides):
            raise ValueError("Length of 'dims' and 'strides' must be equal.")
        self.method = method
        self.dims = dims
        self.strides = strides
        self.kwargs = kwargs

    def fit(self, X: xr.DataArray, y=None):
        """Fit method for compatibility with sklearn's TransformerMixin."""
        return self

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
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
        coarsen_dims = {dim: stride for dim, stride in zip(self.dims, self.strides)}
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
            dim: slice(None, None, stride)
            for dim, stride in zip(self.dims, self.strides)
        }
        return array.isel(indexers)
