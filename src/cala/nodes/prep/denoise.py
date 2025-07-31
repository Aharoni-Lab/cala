from collections.abc import Callable
from typing import Annotated as A
from typing import Any, Literal

import cv2
import numpy as np
import xarray as xr
from noob import Name

from cala.assets import Frame


def denoise(
    frame: Frame, method: Literal["gaussian", "median", "bilateral"] = "gaussian", **kwargs: Any
) -> A[Frame, Name("frame")]:
    """Denoise a single frame."""
    methods: dict[str, Callable] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    _func = methods[method]
    frame = frame.array

    denoised = _func(frame.values.astype(np.float32), **kwargs).astype(np.float64)

    return Frame.from_array(xr.DataArray(denoised, dims=frame.dims, coords=frame.coords))
