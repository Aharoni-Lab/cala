# adj_corr filt_fft local_extreme med_baseline

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import median_filter


def local_extreme(image: np.ndarray, selem: np.ndarray, diff_threshold: int | float) -> np.ndarray:
    """
    Find local maxima in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    selem : np.ndarray
        Structuring element used to define neighbourhood.
    diff_threshold : Union[int, float]
        Minimum difference between local maximum and its neighbours.

    Returns
    -------
    maxima : np.ndarray
        Binary image with 1 at local maxima positions.
    """
    local_max = ndi.maximum_filter(image, footprint=selem) == image
    background = image == 0
    eroded_background = ndi.maximum_filter(background, footprint=selem)
    detected_peaks = local_max ^ eroded_background

    if diff_threshold > 0:
        dilated = ndi.maximum_filter(image, footprint=selem)
        difference = dilated - image
        detected_peaks = np.logical_and(detected_peaks, difference >= diff_threshold)

    return detected_peaks.astype(np.uint8)


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
