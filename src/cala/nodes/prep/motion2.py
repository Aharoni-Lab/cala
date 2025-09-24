import functools
from typing import Callable

import cv2
import numpy as np
import xarray as xr
from noob import process_method
from numpy.fft import ifftshift
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr, Field
from skimage.filters import difference_of_gaussians

from cala.assets import Frame
from cala.models import AXIS
from cala.testing.util import shift_by


class Shift(BaseModel):
    height: float
    width: float

    @classmethod
    def from_arr(cls, array: NDArray) -> "Shift":
        assert array.shape == (2,)
        return Shift(height=array[1], width=array[0])

    def __add__(self, other: "Shift") -> "Shift":
        return Shift(height=self.height + other.height, width=self.width + other.width)


class LockOn(BaseModel):
    max_shifts: tuple[float, float] = (50, 50)
    upsample_factor: int = 10

    prep_kwargs: dict = Field(default_factory=dict)

    _reg_shift: Callable = PrivateAttr(None)
    """A callable used to find the shift"""
    _local: xr.DataArray = PrivateAttr(None)
    """local anchor - processed and ready for comparison"""
    _anchor: xr.DataArray = PrivateAttr(None)
    """global anchor - processed and ready for comparison"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @process_method
    def stabilize(self, frame: Frame) -> Frame:
        arr = frame.array
        prepped = prepare(arr)
        if not self._has_prereqs:
            self._init(prepped)
            return frame

        # --- image, prepped, local --- #
        # image: original image. only shifted and outputted
        # prepped: processed image. only used to find the shift and then discarded
        # local: shifted prepped from the last iteration to be used as a template

        # Steps:
        # 1. raw prepped gets shifted to last local anchor. We save the shift
        # 2. the shifted prepped gets shifted to global anchor. We add to the shift
        # 3. we apply total shift to image. We also save the total shifted prepped
        #    as the anchor for the next frame

        total = Shift(height=0, width=0)
        for template in [self._local, self._anchor]:
            shift_arr, _, _ = self._reg_shift(prepped.values, template.values)
            shift = Shift.from_arr(shift_arr)
            total += shift
            prepped = apply_shift(prepped, shift)
        self._get_ready_for_next(prepped)

        result = apply_shift(arr, total)
        return Frame.from_array(result)

    @property
    def _has_prereqs(self) -> bool:
        return self._local is not None

    def _init(self, image: xr.DataArray) -> None:
        self._local = image
        self._anchor = image
        self._reg_shift = functools.partial(
            register_shift, upsample_factor=self.upsample_factor, max_shifts=self.max_shifts
        )

    def _calculate_shift(self, shift: xr.DataArray) -> xr.DataArray: ...

    def _get_ready_for_next(self, prepped: xr.DataArray) -> None:
        self._local = prepped

        # global learns the local
        curr_idx = prepped[AXIS.frame_coord].item()
        self._anchor = (self._anchor * curr_idx + self._local) / (curr_idx + 1)


def prepare(image: xr.DataArray) -> xr.DataArray:
    tmp = difference_of_gaussians(image, low_sigma=3)
    tmp = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.GaussianBlur(tmp.astype(float), (11, 11), 20)
    return xr.DataArray(result, dims=image.dims, coords=image.coords)


def register_shift(
    src_image: np.ndarray,
    target_image: np.ndarray,
    max_shifts: tuple[float, float] = (20, 20),
    upsample_factor: int = 1,
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
       [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
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

    new_cross_corr[max_shifts[0] : -max_shifts[0], :] = 0
    new_cross_corr[:, max_shifts[1] : -max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size // 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = src_freq.size * upsample_factor**2
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(
            image_product.conj(), upsampled_region_size, upsample_factor, sample_region_offset
        ).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(
            np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape),
            dtype=np.float64,
        )
        maxima -= dftshift
        shifts = shifts + (maxima / upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(), 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(), 1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input data's number of dimensions."
            )

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor))
        * (ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(data.shape[1] // 2)).dot(
            np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1]
        )
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor))
        * (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] - np.floor(data.shape[0] // 2)
        )
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
            (-1j * 2 * np.pi / (data.shape[2] * upsample_factor))
            * (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] - np.floor(data.shape[2] // 2)
            )
        )

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes=[1, 0])
    output = np.tensordot(output, col_kernel, axes=[1, 0])

    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes=[1, 1])
    # output = row_kernel.dot(data).dot(col_kernel)
    return output


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def apply_shifts_dft(src_freq, shifts, diffphase, is_freq: bool = True):
    """
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    if not is_freq:

        src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
        src_freq = cv2.dft(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
        src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    nr, nc = src_freq.shape
    Nr = ifftshift(np.arange(-np.fix(nr / 2.0), np.ceil(nr / 2.0)))
    Nc = ifftshift(np.arange(-np.fix(nc / 2.0), np.ceil(nc / 2.0)))
    Nc, Nr = np.meshgrid(Nc, Nr)
    Greg = src_freq * np.exp(1j * 2 * np.pi * (-shifts[0] * Nr / nr - shifts[1] * Nc / nc))

    Greg = Greg.dot(np.exp(1j * diffphase))
    Greg = np.dstack([np.real(Greg), np.imag(Greg)])
    new_img = cv2.idft(Greg)[:, :, 0]

    max_w, max_h, min_w, min_h = 0, 0, 0, 0
    max_h, max_w = np.ceil(np.maximum((max_h, max_w), shifts[:2])).astype(int)
    min_h, min_w = np.floor(np.minimum((min_h, min_w), shifts[:2])).astype(int)

    new_img[:max_h] = new_img[max_h]
    if min_h < 0:
        new_img[min_h:] = new_img[min_h - 1]
    if max_w > 0:
        new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
    if min_w < 0:
        new_img[:, min_w:] = new_img[:, min_w - 1, np.newaxis]

    return new_img


def apply_shift(image: xr.DataArray, shift: Shift) -> xr.DataArray:
    M = np.float32([[1, 0, shift.width], [0, 1, shift.height]])

    shifted_frame = cv2.warpAffine(
        image.values,
        M,
        (image.sizes[AXIS.width_dim], image.sizes[AXIS.height_dim]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return xr.DataArray(shifted_frame, dims=image.dims, coords=image.coords)


def check_shift_validity(
    source: xr.DataArray, target: xr.DataArray, shift: np.ndarray, threshold: float
) -> bool:
    expected = np.array([5, 5])
    tester = shift_by(source.values, *expected)
    total, _, _ = register_shift(tester, target.values)
    result = total - shift
    error = np.linalg.norm(result - expected)
    return error < threshold
