# let's test flattening against various images

import cv2
import numpy as np
import pytest
from numpydantic.ndarray import NDArray
from skimage.filters import butterworth, difference_of_gaussians
from skimage.registration import phase_cross_correlation

from cala.config import config
from cala.nodes.io import stream


@pytest.fixture
def real() -> NDArray:
    return stream(["cala/msCam1.avi", "cala/msCam2.avi", "cala/msCam3.avi", "cala/msCam4.avi"])


def shift(img: np.ndarray, right_pix: int, down_pix: int) -> NDArray:
    M = np.float32([[1, 0, right_pix], [0, 1, down_pix]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def remove_mean(img: NDArray) -> NDArray:
    """median does not remove the line noises"""
    tmp = img - np.mean(img, axis=0, keepdims=True)
    return tmp - np.mean(tmp, axis=1, keepdims=True)


def prep(img: NDArray) -> NDArray:
    img = butterworth(img)
    # img = _remove_lines(img)
    # img = _remove_lines(img.T).T
    img = remove_mean(img)
    # img = img[100:200, 600:700]
    img = difference_of_gaussians(img, low_sigma=3)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
    # img = normalize(img)
    # img = cv2.medianBlur(img, 11)
    # img = img.astype(int) - np.median(img)
    # img = np.abs(img)
    # img[img < np.quantile(img, 0.75)] = 0
    # img = cv2.GaussianBlur(img, (11, 11), 20)
    # img = difference_of_gaussians(img, low_sigma=4)

    return img


def normalize(img: NDArray) -> NDArray:
    img = (img - img.mean()) / img.std()
    return img


def test_real(real):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(config.video_dir / "mc_test.avi", fourcc, 24.0, (752, 960))

    for i, fr_curr in enumerate(real):
        fr_curr = cv2.medianBlur(fr_curr, 3)

        if i == 199:
            fr_prev = fr_curr

        if i == 200:
            patch_prev = prep(fr_prev)
            patch_curr = prep(fr_curr)
            patch_test = prep(shift(fr_curr, 5, 5))

            real_drift, _, _ = phase_cross_correlation(patch_prev, patch_curr)
            test_drift, _, _ = phase_cross_correlation(patch_prev, patch_test)
            test_check, _, _ = phase_cross_correlation(patch_curr, patch_test)

            acid_drift = register_translation(patch_prev.astype(float), patch_curr.astype(float))
            acid_check = register_translation(patch_prev.astype(float), patch_test.astype(float))
            import cProfile

            cProfile.run("register_translation(patch_prev.astype(float), patch_curr.astype(float))")
            M = np.float32([[1, 0, acid_drift[1]], [0, 1, acid_drift[0]]])

            fr_corr = cv2.warpAffine(
                fr_curr,
                M,
                (fr_curr.shape[1], fr_curr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            fr_prev = fr_corr
            patch_corr = prep(fr_corr)[100:200, 600:700]

            frame_bgr = cv2.cvtColor(np.concat([fr_curr, fr_corr]), cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)


def register_translation(
    src_image,
    target_image,
    shifts_lb=None,
    shifts_ub=None,
    max_shifts=(10, 10),
):
    """
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """

    src_freq_1 = cv2.dft(src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
    src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
    src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
    target_freq_1 = cv2.dft(target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
    target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
    target_freq = np.array(target_freq, dtype=np.complex128, copy=False)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    image_product_cv = np.dstack([np.real(image_product), np.imag(image_product)])
    cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
    cross_correlation = cross_correlation[:, :, 0] + 1j * cross_correlation[:, :, 1]

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):
        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0] : shifts_lb[0], :] = 0
        else:
            new_cross_corr[: shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0] :, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1] : shifts_lb[1]] = 0
        else:
            new_cross_corr[:, : shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1] :] = 0
    else:
        new_cross_corr[max_shifts[0] : -max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1] : -max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size // 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts  # , src_freq
