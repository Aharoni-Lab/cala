from dataclasses import dataclass
from typing import Dict, Optional, Literal

import cv2
import numpy as np
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
    clip_negative: bool = True
    """Whether to clip negative values after background removal."""

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

        # Store parameters as attributes for easy access
        self.method = self.params.method
        self.kernel_size = self.params.kernel_size
        self.clip_negative = self.params.clip_negative

        # Pre-compute kernel for tophat method
        if self.method == "tophat":
            self.kernel = disk(self.kernel_size)

    def learn_one(self, frame: np.ndarray) -> "BackgroundEraser":
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

    def transform_one(self, frame: np.ndarray) -> np.ndarray:
        """Remove background from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to process

        Returns
        -------
        np.ndarray
            Frame with background removed
        """
        frame = frame.astype(np.float32)

        if self.method == "uniform":
            # Estimate background using uniform filter
            background = uniform_filter(frame, size=self.kernel_size)
            result = frame - background
        else:  # tophat
            # Apply morphological tophat operation
            result = cv2.morphologyEx(
                frame, cv2.MORPH_TOPHAT, self.kernel.astype(np.uint8)
            )

        if self.clip_negative:
            result = np.maximum(result, 0)

        return result

    def get_info(self) -> Dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current parameters
        """
        return {
            "method": self.method,
            "kernel_size": self.kernel_size,
            "clip_negative": self.clip_negative,
        }
