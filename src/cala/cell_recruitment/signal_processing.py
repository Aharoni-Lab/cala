# adj_corr filt_fft local_extreme med_baseline
from typing import Union
import numpy as np
import scipy.ndimage as ndi


def local_extreme(
    image: np.ndarray, selem: np.ndarray, diff_threshold: Union[int, float]
) -> np.ndarray:
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
