from collections.abc import Callable
from typing import Annotated as A
from typing import Any, ClassVar, Literal

import cv2
import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from pydantic import Field, PrivateAttr

from cala.models import Frame


class Denoiser(Node):
    """Streaming denoiser for calcium imaging data.

    This transformer applies denoising to each frame using OpenCV methods:
    - Gaussian blur
    - Median blur
    - Bilateral filter
    """

    method: Literal["gaussian", "median", "bilateral"] = "gaussian"
    """one of ['gaussian', 'median', 'bilateral']"""
    kwargs: dict = Field(default_factory=dict)
    """kwargs for the denoising method"""

    METHODS: ClassVar[dict[str, Callable]] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }
    _func: Callable = PrivateAttr(init=False)

    def model_post_init(self, __context: None = None) -> None:
        """Initialize the denoiser with given parameters."""
        self._func = self.METHODS[self.method]

    def process(self, frame: Frame) -> A[Frame, Name("frame")]:
        """Denoise a single frame.

        Parameters
        ----------
        frame : Frame
            Input frame to denoise

        Returns
        -------
        Frame
            Denoised frame
        """
        frame = frame.array.astype(np.float32)

        denoised = self._func(frame.values, **self.kwargs)

        return Frame(array=xr.DataArray(denoised, dims=frame.dims, coords=frame.coords))


def denoise(
    frame: Frame, method: Literal["gaussian", "median", "bilateral"] = "gaussian", **kwargs: Any
) -> A[Frame, Name("frame")]:
    """Denoise a single frame."""
    METHODS: dict[str, Callable] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    _func = METHODS[method]
    frame = frame.array

    denoised = _func(frame.values, **kwargs)

    return Frame(array=xr.DataArray(denoised, dims=frame.dims, coords=frame.coords))
