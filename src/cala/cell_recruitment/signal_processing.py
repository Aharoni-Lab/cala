# adj_corr filt_fft local_extreme med_baseline
from typing import Union, Literal
import numpy as np
import cv2
from scipy.ndimage.filters import median_filter


def local_extreme(
    image: np.ndarray,
    selem: np.ndarray,
    intensity_threshold: Union[int, float],
    mode: Literal["max", "min"],
) -> np.ndarray:
    """
    Find local maxima in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    selem : np.ndarray
        Structuring element used to define neighbourhood.
    intensity_threshold : Union[int, float]
        Minimum difference between local maximum and its neighbours.
    mode: Literal["max", "min"]
        Local max vs. local min.

    Returns
    -------
    maxima : np.ndarray
        Binary image with 1 at local maxima positions.
    """
    # Compute the morphological gradient (difference between dilation and erosion)
    image_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, selem)
    diff_mask = image_gradient > intensity_threshold

    if mode == "max":
        image_max = cv2.dilate(image, selem)
        extrema_mask = image == image_max
    elif mode == "min":
        image_min = cv2.erode(image, selem)
        extrema_mask = image == image_min
    else:
        raise ValueError(f"Invalid mode '{mode}'; expected 'min' or 'max'.")

    # Combine the extrema mask with the difference mask
    image_extrema = np.logical_and(extrema_mask, diff_mask)

    return image_extrema.astype(np.uint8)


def median_clipper(a: np.ndarray, window_size: int) -> np.ndarray:
    """
    Subtract baseline from a timeseries as estimated by median-filtering the
    timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    window_size : int
        Window size of the median filter. This parameter is passed as `size` to
        :func:`scipy.ndimage.filters.median_filter`.

    Returns
    -------
    a : np.ndarray
        Timeseries with baseline subtracted.
    """
    base = median_filter(a, size=window_size)
    a -= base
    return a.clip(0, None)
