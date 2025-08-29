from typing import Annotated as A
from typing import Any

import xarray as xr
from noob import Name
from skimage.filters import butterworth
from skimage.restoration import rolling_ball

from cala.assets import Frame


def butter(frame: Frame, kwargs: dict[str, Any]) -> A[Frame, Name("frame")]:
    """
    butterworth filter centers the image to zero. this causes two images with same intensity ratio
    across pixels to be indistinguishable.
    To recover the absolute brightness, we shift the filtered image by the
    mean brightness of the original frame.
    """
    arr = butterworth(frame.array, **kwargs) + frame.array.mean().item()

    return Frame.from_array(xr.DataArray(arr, dims=frame.array.dims, coords=frame.array.coords))


def ball(frame: Frame, kwargs: dict[str, Any]) -> Frame:
    """
    takes a VERY long time. also not as good as butterworth at handling clustered cells (all bright
    region)
    """
    bg = rolling_ball(frame.array, **kwargs)
    frame.array -= bg

    return frame
