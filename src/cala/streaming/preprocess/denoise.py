from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Self

import cv2
import numpy as np
import xarray as xr
from river import base

from cala.streaming.core import Parameters


@dataclass
class DenoiserParams(Parameters):
    """Denoiser parameters"""

    method: str = "gaussian"
    """one of ['gaussian', 'median', 'bilateral']"""
    kwargs: dict = field(default_factory=dict)
    """kwargs for the denoising method"""

    def validate(self) -> None:
        """Validate denoising parameters"""
        if self.method not in Denoiser.METHODS:
            raise ValueError(
                f"denoise method '{self.method}' not understood. "
                f"Available methods are: {', '.join(Denoiser.METHODS.keys())}"
            )


@dataclass
class Denoiser(base.Transformer):
    """Streaming denoiser for calcium imaging data.

    This transformer applies denoising to each frame using OpenCV methods:
    - Gaussian blur
    - Median blur
    - Bilateral filter
    """

    params: DenoiserParams = field(default_factory=DenoiserParams)

    METHODS: ClassVar[dict[str, Callable]] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    def __post_init__(self) -> None:
        """Initialize the denoiser with given parameters."""
        self.func = self.METHODS[self.params.method]

    def learn_one(self, frame: xr.DataArray) -> Self:
        """Update statistics from new frame.

        Parameters
        ----------
        frame : xr.DataArray
            Input frame to learn from

        Returns
        -------
        self : Denoiser
            The denoiser instance
        """
        return self

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        """Denoise a single frame.

        Parameters
        ----------
        frame : xr.DataArray
            Input frame to denoise

        Returns
        -------
        xr.DataArray
            Denoised frame
        """
        frame = frame.astype(np.float32)

        denoised = self.func(frame.values, **self.params.kwargs)

        return xr.DataArray(denoised, dims=frame.dims, coords=frame.coords)

    def get_info(self) -> dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current statistics
        """
        return {
            "method": self.params.method,
            "func": self.func.__name__,
            "kwargs": self.params.kwargs,
        }
