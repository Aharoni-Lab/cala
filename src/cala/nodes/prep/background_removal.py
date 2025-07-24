from typing import Annotated as A
from typing import Literal

import cv2
import numpy as np
import xarray as xr
from noob import Name
from scipy.ndimage import uniform_filter
from skimage.morphology import disk

from cala.models import Frame


def remove_background(
    frame: Frame, method: Literal["uniform", "tophat"] = "uniform", kernel_size: int = 3
) -> A[Frame, Name("frame")]:
    """Streaming transformer that removes background from video frames.

    This transformer implements online background removal using either:
    1. Uniform filtering - Background is estimated by convolving each frame with a uniform kernel
    2. Tophat - Morphological tophat operation using a disk-shaped kernel

    method:
        - "uniform": Use uniform filtering to estimate background
        - "tophat": Use morphological tophat operation to estimate background
    kernel_size:
        Size of the kernel for background removal.
        for "uniform", bigger kernel_size removes more background.
        for "tophat", bigger kernel_size preserves more background.
    """
    frame = frame.array.astype(np.float32)

    if method == "uniform":
        # Estimate background using uniform filter
        background = uniform_filter(frame.values, size=kernel_size)
        result = frame.where(frame <= background, other=0).values
    elif method == "tophat":
        # Apply morphological tophat operation
        kernel = disk(kernel_size)
        result = cv2.morphologyEx(frame.values, cv2.MORPH_TOPHAT, kernel.astype(np.uint8))
    else:
        raise NotImplementedError(f"Unknown method {method}")

    return Frame(
        array=xr.DataArray(result.astype(np.float64), dims=frame.dims, coords=frame.coords)
    )
