from dataclasses import dataclass, field
from typing import Dict, Optional, Self

import cv2
import numpy as np
from river import base

from ..core import Parameters


@dataclass
class DenoiserParams(Parameters):
    """Denoiser parameters"""

    method: str = "gaussian"
    """one of ['gaussian', 'median', 'bilateral']"""
    kwargs: dict = field(default_factory=dict)
    """kwargs for the denoising method"""

    def _validate_parameters(self) -> None:
        """Validate denoising parameters"""
        if self.method not in Denoiser.METHODS:
            raise ValueError(
                f"denoise method '{self.method}' not understood. "
                f"Available methods are: {', '.join(Denoiser.METHODS.keys())}"
            )


class Denoiser(base.Transformer):
    """Streaming denoiser for calcium imaging data.

    This transformer applies denoising to each frame using OpenCV methods:
    - Gaussian blur
    - Median blur
    - Bilateral filter
    """

    METHODS = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    def __init__(self, params: Optional[DenoiserParams] = None):
        """Initialize the denoiser with given parameters."""
        super().__init__()

        self.params = params or DenoiserParams()
        self.func = self.METHODS[self.params.method]

    def learn_one(self, frame: np.ndarray) -> Self:
        """Update statistics from new frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to learn from

        Returns
        -------
        self : Denoiser
            The denoiser instance
        """
        return self

    def transform_one(self, frame: np.ndarray) -> np.ndarray:
        """Denoise a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to denoise

        Returns
        -------
        np.ndarray
            Denoised frame
        """
        frame = frame.astype(np.float32)

        denoised = self.func(frame, **self.params.kwargs)

        return denoised

    def get_info(self) -> Dict:
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
