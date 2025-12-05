from typing import Annotated as A
from typing import Any

import xarray as xr
from noob import Name
from skimage.filters import butterworth
from skimage.restoration import rolling_ball

from cala.arrays import Frame


def butter(frame: Frame, kwargs: dict[str, Any]) -> A[Frame, Name("frame")]:
    """
    Butterworth filter centers the image to zero. This is due to the constant term (the mean)
    being expressed as the 0th term in the fourier series.
    Since the absolute background activity does not matter (all that is left is the high-frequency
    signal), we simply add half of the 8-bit pixel max so that the total cannot exceed the
    0-255 range.

    The filter can also be used to reduce the scattering and the glow! (inspired by Marcel Brosche)
    This helps remove overlap between cells (with higher cutoff_frequency_ratio)
    """
    arr = butterworth(frame.array, **kwargs) + 2**7

    return Frame.from_array(xr.DataArray(arr, dims=frame.array.dims, coords=frame.array.coords))


def ball(frame: Frame, kwargs: dict[str, Any]) -> Frame:
    """
    takes a VERY long time. also not as good as butterworth at handling clustered cells (all bright
    region)
    """
    bg = rolling_ball(frame.array, **kwargs)
    frame.array -= bg

    return frame
