from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import xarray as xr
from river import base

from cala.streaming.core import Parameters


@dataclass
class DownsamplerParams(Parameters):
    """Downsampler parameters"""

    method: Literal["mean", "subset"] = "mean"
    """The downsampling method to use ('mean' or 'subset')."""
    dimensions: list[str] = field(default_factory=lambda: ["width", "height"])
    """The dimensions along which to downsample."""
    strides: list[int] = field(default_factory=lambda: [1, 1])
    """The strides or pool sizes for each dimension."""
    kwargs: dict = field(default_factory=dict)
    """keyword arguments for each downsampling method"""

    def validate(self) -> None:
        if self.method not in ("mean", "subset"):
            raise ValueError(f"Downsampling method '{self.method}' not understood.")
        if len(self.dimensions) != len(self.strides):
            raise ValueError("Length of 'dims' and 'strides' must be equal.")
        if any(i < 1 for i in self.strides):
            raise ValueError("'strides' must be greater than 1.")


@dataclass
class Downsampler(base.Transformer):
    """Streaming downsampler for calcium imaging data.

    This transformer applies downsample to each frame using:
    - subset
    - mean

    Attributes
    _frame_number (int): The current frame number. Initiates with -1 so that we can immediately add one
                         and start with 0.

    """

    params: DownsamplerParams = field(default_factory=DownsamplerParams)

    def learn_one(self, frame: xr.DataArray) -> Self:
        """Update statistics from new frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to learn from

        Returns
        -------
        self : Downsampler
            The downsampler instance
        """
        return self

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray | None:
        """Downsample a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to downsample

        Returns
        -------
        np.ndarray
            Downsampled frame
        """
        frame = frame.astype(np.float32)

        if self.params.method == "mean":
            return self.mean_downsample(frame)
        elif self.params.method == "subset":
            return self.subset_downsample(frame)

    def get_info(self) -> dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current statistics
        """
        return {
            "method": self.params.method,
            "dimensions": self.params.dimensions,
            "strides": self.params.strides,
            "kwargs": self.params.kwargs,
        }

    def mean_downsample(self, array: xr.DataArray) -> xr.DataArray:
        """
        Downsample the array by taking the mean over specified window sizes.

        Parameters:
            array (xr.DataArray): The DataArray to downsample.

        Returns:
            xr.DataArray: The downsampled DataArray.
        """
        coarsen_dims = {
            dim: stride for dim, stride in zip(self.params.dimensions, self.params.strides)
        }
        return array.coarsen(coarsen_dims, boundary="trim").mean(**self.params.kwargs)

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
            for dim, stride in zip(self.params.dimensions, self.params.strides)
        }
        return array.isel(indexers)
