from typing import Annotated as A
import numpy as np

from noob import Name
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch

from cala.assets import Frame


def remove(
    frame: Frame, distortion_freq: float | None = None, num_taps: int = 65, eps: float = 0.025
) -> A[Frame, Name("frame")]:
    arr = frame.array

    if np.all(frame.array == 0):
        return frame

    denoised = _remove_lines(
        arr.values, distortion_freq=distortion_freq, num_taps=num_taps, eps=eps
    )

    dmin = denoised.min()
    if dmin < 0:
        denoised -= dmin

    arr.values = denoised

    return Frame.from_array(arr)


def _remove_lines(image, distortion_freq: float = None, num_taps: int = 65, eps: float = 0.025):
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


def _estimate_distortion_freq(image, min_frequency=1 / 25):
    """Estimates distortion frequency as spectral peak in vertical dim."""
    f, pxx = welch(image.sum(axis=1))
    pxx[f < min_frequency] = 0.0
    return f[pxx.argmax()]
