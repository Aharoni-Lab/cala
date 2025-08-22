from collections.abc import Callable
from typing import Annotated as A
from typing import Any, Literal

import cv2
import numpy as np
import xarray as xr
from noob import Name

from cala.assets import Frame


def denoise(
    frame: Frame, method: Literal["gaussian", "median", "bilateral"], kwargs: dict[str, Any]
) -> A[Frame, Name("frame")]:
    """Denoise a single frame."""
    methods: dict[str, Callable] = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
        "nonlocal": cv2.fastNlMeansDenoising,
    }

    _func = methods[method]

    frame = frame.array

    if method == "nonlocal":
        arr = frame.values.astype(np.uint8)
    else:
        arr = frame.values.astype(np.float32)

    denoised = _func(arr, **kwargs).astype(float)

    return Frame.from_array(xr.DataArray(denoised, dims=frame.dims, coords=frame.coords))
