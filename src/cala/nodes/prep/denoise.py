from collections.abc import Callable
from functools import partial
from typing import Annotated as A
from typing import Any, Literal

import cv2
import numpy as np
import xarray as xr
from noob import Name, process_method
from pydantic import BaseModel
from skimage.restoration import calibrate_denoiser

from cala.assets import Frame


def _bilateral(arr: np.ndarray, **kwargs: Any) -> np.ndarray:
    arr = arr.astype(np.float32)
    return cv2.bilateralFilter(arr, **kwargs)


class Restore(BaseModel):
    kwargs: dict[str, Any] | None = None
    model: Callable = None

    @process_method
    def denoise(self, frame: Frame) -> A[Frame, Name("frame")]:
        arr = frame.array
        if self.model is None:
            if not self.kwargs:
                param_matrix = {
                    "d": list(range(1, 20)),
                    "sigmaColor": [10, 50, 100, 200, 250],
                    "sigmaSpace": [10, 50, 100, 200, 250],
                }
                self.model = calibrate_denoiser(arr, _bilateral, param_matrix)
            else:
                self.model = partial(cv2.bilateralFilter, **self.kwargs)

        denoised = self.model(arr)
        return Frame.from_array(xr.DataArray(denoised, dims=arr.dims, coords=arr.coords))


def blur(
    frame: Frame,
    method: Literal["gaussian", "median", "bilateral", "nonlocal"],
    kwargs: dict[str, Any],
) -> A[Frame, Name("frame")]:
    """Denoise a single frame."""
    methods: dict[str, Callable] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
        "nonlocal": cv2.fastNlMeansDenoising,  # really slow. ~40 ms.
    }

    _func = methods[method]

    frame = frame.array

    arr = frame.values.astype(np.uint8) if method == "nonlocal" else frame.values.astype(np.float32)

    denoised = _func(arr, **kwargs).astype(float)

    return Frame.from_array(xr.DataArray(denoised, dims=frame.dims, coords=frame.coords))
