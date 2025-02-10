from dataclasses import dataclass
from typing import Dict, Optional, Literal

import cv2
import numpy as np
import xarray as xr
from river import base
from scipy.ndimage import uniform_filter
from skimage.morphology import disk

from ..core import Parameters


@dataclass
class BackgroundEraserParams(Parameters):
    """Parameters for background eraser."""

    method: Literal["uniform", "tophat"] = "uniform"
    """Method to use for background removal.
    
    Options:
        - "uniform": Use uniform filtering to estimate background
        - "tophat": Use morphological tophat operation to estimate background
    """
    kernel_size: int = 3
    """Size of the kernel for background removal."""

    def _validate_parameters(self) -> None:
        if self.method not in ["uniform", "tophat"]:
            raise ValueError("method must be one of ['uniform', 'tophat']")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be greater than zero")


class BackgroundEraser(base.Transformer):
    """Streaming transformer that removes background from video frames.

    This transformer implements online background removal using either:
    1. Uniform filtering - Background is estimated by convolving each frame with a uniform kernel
    2. Tophat - Morphological tophat operation using a disk-shaped kernel
    """

    def __init__(self, params: Optional[BackgroundEraserParams] = None):
        """Initialize the background eraser with given parameters."""
        super().__init__()

        # Set default parameters if none provided
        self.params = params or BackgroundEraserParams()

        # Pre-compute kernel for tophat method
        if self.params.method == "tophat":
            self.kernel = disk(self.params.kernel_size)

    def learn_one(self, frame: xr.DataArray) -> "BackgroundEraser":
        """Update any learning parameters with new frame.

        This transformer doesn't need to learn from the data, so this is a no-op.

        Parameters
        ----------
        frame : np.ndarray
            Input frame

        Returns
        -------
        self : BackgroundEraser
            The transformer instance
        """
        return self

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        """Remove background from a single frame.

        Parameters
        ----------
        frame : xr.DataArray
            Input frame to process

        Returns
        -------
        xr.DataArray
            Frame with background removed
        """
        frame = frame.astype(np.float32)

        if self.params.method == "uniform":
            # Estimate background using uniform filter
            background = uniform_filter(frame.values, size=self.params.kernel_size)
            result = frame.values - background
        else:  # tophat
            # Apply morphological tophat operation
            result = cv2.morphologyEx(
                frame.values, cv2.MORPH_TOPHAT, self.kernel.astype(np.uint8)
            )

        result[result < 0] = 0

        return xr.DataArray(result, dims=frame.dims, coords=frame.coords)

    def get_info(self) -> Dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current parameters
        """
        return {
            "method": self.params.method,
            "kernel_size": self.params.kernel_size,
        }
