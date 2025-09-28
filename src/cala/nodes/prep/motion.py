import functools
from collections.abc import Callable
from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name, process_method
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
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
        return Shift(height=array[0], width=array[1])

    def __add__(self, other: "Shift") -> "Shift":
        return Shift(height=self.height + other.height, width=self.width + other.width)


class Anchor(BaseModel):
    max_shift_w: int = 20
    max_shift_h: int = 20

    dog_kwargs: dict = Field(default_factory=dict, validate_default=True)
    gauss_kwargs: dict = Field(default_factory=dict, validate_default=True)

    _reg_shift: Callable = PrivateAttr(None)
    """A callable used to find the shift"""
    _local: xr.DataArray = PrivateAttr(None)
    """local anchor - processed and ready for comparison"""
    _global: xr.DataArray = PrivateAttr(None)
    """global anchor - processed and ready for comparison"""
    _history: list[Shift] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("dog_kwargs", mode="before")
    @classmethod
    def default_dog(cls, value: dict) -> dict:
        if not value:
            return {"low_sigma": 3}
        else:
            return value

    @field_validator("gauss_kwargs", mode="before")
    @classmethod
    def default_gauss(cls, value: dict) -> dict:
        if not value:
            return {"ksize": (11, 11), "sigmaX": 20}
        else:
            return value

    @process_method
    def stabilize(self, frame: Frame) -> A[Frame, Name("frame")]:
        """
        --- image, prepped, local ---
        image: original image. only shifted and outputted
        prepped: processed image. only used to find the shift and then discarded
        local: shifted prepped from the last iteration to be used as a template

        Steps:
        1. raw prepped gets shifted to last local anchor. We save the shift
        2. the shifted prepped gets shifted to global anchor. We add to the shift
        3. we apply total shift to image. We also save the total shifted prepped
           as the anchor for the next frame
        """
        arr = frame.array
        prepped = prepare(arr, dog_kwargs=self.dog_kwargs, gauss_kwargs=self.gauss_kwargs)
        if not self._has_prereqs:
            self._init(prepped)
            return frame

        total = Shift(height=0, width=0)
        for template in [self._local, self._global]:
            shift_arr = match_template(prepped.values, template.values)
            # shift_arr, _, _ = self._reg_shift(template.values, prepped.values)
            shift = Shift.from_arr(shift_arr)
            total += shift
            prepped = apply_shift(prepped, shift)
        self._get_ready_for_next(prepped)
        self._history.append(total)

        result = apply_shift(arr, total)
        return Frame.from_array(result)

    @property
    def _has_prereqs(self) -> bool:
        return self._local is not None

    def _init(self, image: xr.DataArray) -> None:
        self._local = image
        self._global = image
        self._reg_shift = functools.partial(
            match_template, max_shift_w=self.max_shift_w, max_shift_h=self.max_shift_h
        )

    def _calculate_shift(self, shift: xr.DataArray) -> xr.DataArray: ...

    def _get_ready_for_next(self, prepped: xr.DataArray) -> None:
        self._local = prepped

        # global learns the local
        curr_idx = prepped[AXIS.frame_coord].item()
        self._global = (self._global * curr_idx + self._local) / (curr_idx + 1)


def prepare(image: xr.DataArray, dog_kwargs: dict, gauss_kwargs: dict) -> xr.DataArray:
    tmp = difference_of_gaussians(image, **dog_kwargs)
    tmp = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.GaussianBlur(tmp.astype(float), **gauss_kwargs)
    return xr.DataArray(result, dims=image.dims, coords=image.coords)


def match_template(
    image: np.ndarray, template: np.ndarray, max_shift_w: int = 10, max_shift_h: int = 10
) -> np.ndarray:
    """https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html"""
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w

    templ_crop = template[max_shift_h : h_i - max_shift_h, max_shift_w : w_i - max_shift_w].astype(
        np.float32
    )

    res = cv2.matchTemplate(image, templ_crop, cv2.TM_CCORR_NORMED)
    top_left = cv2.minMaxLoc(res)[3]

    sh_y, sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x - 1, sh_y])
        log_xp1_y = np.log(res[sh_x + 1, sh_y])
        log_x_ym1 = np.log(res[sh_x, sh_y - 1])
        log_x_yp1 = np.log(res[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(res[sh_x, sh_y])

        sh_x_n = -(
            sh_x - ms_h + ((log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        )
        sh_y_n = -(
            sh_y - ms_w + ((log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
        )
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)

    shift = np.array([sh_x_n, sh_y_n])

    return shift


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
    source: xr.DataArray, target: xr.DataArray, shift: np.ndarray, threshold: float, func: Callable
) -> bool:
    expected = np.array([5, 5])
    tester = shift_by(source.values, *expected)
    total = func(tester, target.values)
    result = total - shift
    error = np.linalg.norm(result - expected)
    return error < threshold
