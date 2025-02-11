from dataclasses import dataclass, field
from typing import Optional, Self

import cv2
import numpy as np
import xarray as xr
from river import base
from skimage.registration import phase_cross_correlation

from ..core import Parameters


@dataclass
class RigidTranslatorParams(Parameters):
    max_shift: Optional[int] = None
    kwargs: dict = field(default_factory=dict)

    def _validate_parameters(self) -> None:
        if self.max_shift is not None and self.max_shift < 0:
            raise ValueError("max_shift must be a positive integer.")


@dataclass
class RigidTranslator(base.Transformer):
    """Handles motion_stabilization correction"""

    params: RigidTranslatorParams
    _learn_count: int = 0
    _transform_count: int = 0
    anchor_frame_: np.ndarray = field(init=False)
    motion_: list = field(default_factory=list)

    def learn_one(self, frame: xr.DataArray) -> Self:
        if not hasattr(self, "anchor_frame_"):
            self.anchor_frame_ = frame.values
            return self

        shift, error, diffphase = phase_cross_correlation(
            self.anchor_frame_, frame.values, **self.params.kwargs
        )
        if self.params.max_shift is not None:
            # Cap shift values at max amplitude
            shift = np.clip(shift, -self.params.max_shift, self.params.max_shift)

        self.motion_.append(shift)  # shift = [shift_y, shift_x]
        self._learn_count += 1
        return self

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        if len(self.motion_) == 0:
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
        self.anchor_frame_ = transformed_frame

        return xr.DataArray(transformed_frame, dims=frame.dims, coords=frame.coords)
