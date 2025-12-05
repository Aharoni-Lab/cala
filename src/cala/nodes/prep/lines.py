from typing import Annotated as A
from typing import Any, Literal

import numpy as np
from noob import Name
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch

from cala.arrays import AXIS, Frame


def remove_mean(frame: Frame, orient: Literal["horiz", "vert", "both"]) -> A[Frame, Name("frame")]:
    arr = frame.array

    if orient == "horiz":
        denoised = arr - arr.mean(dim=AXIS.width_dim)
    elif orient == "vert":
        denoised = arr - arr.mean(dim=AXIS.height_dim)
    elif orient == "both":
        horiz_dn = arr - arr.mean(dim=AXIS.width_dim)
        denoised = horiz_dn - horiz_dn.mean(dim=AXIS.height_dim)
    else:
        raise ValueError(f"Unknown orientation {orient}")

    # diff should be frame.mean - denoised.mean, but denoised.mean is always 0 by definition
    diff = frame.array.mean()

    return Frame.from_array(denoised + diff)


def remove_freq(
    frame: Frame,
    orient: Literal["horiz", "vert", "both"],
    kwargs: dict[str, Any] | None = None,
) -> A[Frame, Name("frame")]:
    if kwargs is None:
        kwargs = {}

    arr = frame.array

    if np.all(frame.array == 0):
        return frame

    if orient == "horiz":
        denoised = _remove_lines(arr.values, **kwargs)
    elif orient == "vert":
        denoised = _remove_lines(arr.values.T, **kwargs).T
    elif orient == "both":
        horiz_dn = _remove_lines(arr.values, **kwargs)
        denoised = _remove_lines(horiz_dn.T, **kwargs).T

    dmin = denoised.min()
    if dmin < 0:
        denoised -= dmin

    arr.values = denoised

    return Frame.from_array(arr)


def _remove_lines(
    image: np.ndarray, distortion_freq: float = None, num_taps: int = 65, eps: float = 0.025
) -> np.ndarray:
    """
    Removes horizontal line artifacts from scanned image.
    Args:
      image: 2D or 3D array.
      distortion_freq: Float, distortion frequency in cycles/pixel, or
        `None` to estimate from spectrum.
      num_taps: Integer, number of filter taps to use in each dimension.
      eps: Small positive param to adjust filters cutoffs (cycles/pixel).
    Returns:
      Denoised image.
    """
    if distortion_freq is None:
        distortion_freq = _estimate_distortion_freq(image)

    hpf = firwin(num_taps, distortion_freq - eps, pass_zero="highpass", fs=1)
    lpf = firwin(num_taps, eps, pass_zero="lowpass", fs=1)
    return image - convolve1d(convolve1d(image, hpf, axis=0), lpf, axis=1)


def _estimate_distortion_freq(image: np.ndarray, min_frequency: float = 1 / 25) -> float:
    """Estimates distortion frequency as spectral peak in vertical dim."""
    f, pxx = welch(image.sum(axis=1))
    pxx[f < min_frequency] = 0.0
    return f[pxx.argmax()]
