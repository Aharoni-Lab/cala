from dataclasses import dataclass, field
from typing import Self

import cv2
import numpy as np
import xarray as xr
from river import base
from skimage.registration import phase_cross_correlation

from cala.streaming.core import Parameters


@dataclass
class RigidStabilizerParams(Parameters):
    drift_speed: float = 1
    anchor_frame_index: int = 0
    kwargs: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.drift_speed is not None and self.drift_speed < 0:
            raise ValueError("drift_speed must be a positive integer.")


@dataclass
class RigidStabilizer(base.Transformer):
    """Handles motion_stabilization correction
    it first registers anchor frame. and then a reference frame.
    the target frame is corrected to reference frame.
    the target frame is then corrected to anchor frame, if the shift between the two is less than max_shift.

    two edge cases:
    the drift magnitude goes over the drift threshold --> gotta use anchor shift
    the anchor shift is bigger than max shift threshold --> gotta use reference shift



    """

    params: RigidStabilizerParams
    _learn_count: int = 0
    _transform_count: int = 0
    _anchor_last_applied_on: int = field(init=False)
    anchor_frame_: np.ndarray = field(init=False)
    previous_frame_: np.ndarray = field(init=False)
    motion_: list = field(default_factory=list)

    def learn_one(self, frame: xr.DataArray) -> Self:
        if not hasattr(self, "anchor_frame_"):
            self.anchor_frame_ = frame.values
            self._anchor_last_applied_on = self._learn_count

        if not hasattr(self, "previous_frame_"):
            self.previous_frame_ = frame.values
            self._learn_count += 1
            return self

        shift, _, _ = phase_cross_correlation(
            self.anchor_frame_, frame.values, **self.params.kwargs
        )

        adjacent_shift, error, diff_phase = phase_cross_correlation(
            self.previous_frame_, frame.values, **self.params.kwargs
        )

        shift_magnitude = np.sqrt(np.sum(np.square(shift)))
        adjacent_shift_magnitude = np.sqrt(np.sum(np.square(adjacent_shift)))

        if (
            shift_magnitude - adjacent_shift_magnitude
            > (self._learn_count - self._anchor_last_applied_on) * self.params.drift_speed
        ):
            shift = adjacent_shift

        else:
            self._anchor_last_applied_on = self._learn_count

        self.motion_.append(shift)  # shift = [shift_y, shift_x]
        self._learn_count += 1
        return self

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        if len(self.motion_) == 0:
            self._transform_count += 1
            return frame

        # Define the affine transformation matrix for translation
        M = np.float32([[1, 0, self.motion_[-1][1]], [0, 1, self.motion_[-1][0]]])

        transformed_frame = cv2.warpAffine(
            frame.values,
            M,
            (frame.shape[1], frame.shape[0]),  # (Width, Height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        np.nan_to_num(transformed_frame, copy=False, nan=0)

        self._transform_count += 1
        self.previous_frame_ = transformed_frame

        return xr.DataArray(transformed_frame, dims=frame.dims, coords=frame.coords)

    def get_info(self) -> dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current statistics
        """
        return {
            "_learn_count": self._learn_count,
            "_transform_count": self._transform_count,
            "_anchor_last_applied_on": self._anchor_last_applied_on,
            "anchor_frame_": self.anchor_frame_,
            "previous_frame_": self.previous_frame_,
            "motion_": self.motion_,
        }
