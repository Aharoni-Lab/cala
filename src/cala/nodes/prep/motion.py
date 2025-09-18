from logging import Logger
from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name, process_method
from pydantic import BaseModel, ConfigDict, Field
from skimage.filters import butterworth
from skimage.registration import phase_cross_correlation

from cala.assets import Frame
from cala.logging import init_logger
from cala.models import AXIS


class Shift(BaseModel):
    width: float
    height: float


class Stabilizer(BaseModel):
    drift_speed: float = 1.0
    patch_size: int = 50
    pcc_kwargs: dict = Field(default_factory=dict)

    filter_kwargs: dict = Field(default_factory=dict)

    _anchor_last_applied_on: int = None
    patch_: dict[str, tuple[int, int]] = None
    anchor_frame_: xr.DataArray = None
    previous_frame_: xr.DataArray = None
    motions_: list[Shift] = None

    logger: Logger = init_logger(__name__)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _interesting_patch(
        frame: np.ndarray, width: int, height: int
    ) -> dict[str, tuple[int, int]]:
        box = [
            {"width": (x, x + width), "height": (y, y + height)}
            for x in range(0, frame.shape[1], width)
            for y in range(0, frame.shape[0], height)
        ]
        return box[
            np.argmax(
                [
                    frame[c["height"][0] : c["height"][1], c["width"][0] : c["width"][1]].var()
                    for c in box
                ]
            )
        ]

    def _get_patch(self, frame: xr.DataArray) -> np.ndarray:
        return frame.values[
            self.patch_["height"][0] : self.patch_["height"][1],
            self.patch_["width"][0] : self.patch_["width"][1],
        ]

    @process_method
    def stabilize(self, frame: Frame) -> A[Frame, Name("frame")]:
        if self._is_first_frame(frame):
            return frame

        curr_frame = frame.array

        shift = self._compute_shift(curr_frame)
        shifted_frame = self._apply_shift(curr_frame, shift)

        self.previous_frame_ = shifted_frame

        if self._anchor_last_applied_on == shifted_frame[AXIS.frame_coord].item():
            self.anchor_frame_ = self._update_anchor(shifted_frame)

        self.motions_.append(shift)

        return Frame.from_array(
            xr.DataArray(shifted_frame, dims=frame.array.dims, coords=frame.array.coords)
        )

    def _is_first_frame(self, frame: Frame) -> bool:
        if (
            (self.anchor_frame_ is not None)
            and (self.previous_frame_ is not None)
            and (self.motions_ is not None)
        ):
            return False

        elif (
            (self.anchor_frame_ is None)
            and (self.previous_frame_ is None)
            and (self.motions_ is None)
        ):
            self._anchor_last_applied_on = 0
            self.patch_ = self._interesting_patch(frame.array, self.patch_size, self.patch_size)
            self.anchor_frame_ = frame.array
            self.previous_frame_ = frame.array
            self.motions_ = [Shift(width=0, height=0)]
            return True

        else:
            raise NotImplementedError(
                f"Undefined State: Only some of the attributes are initialized: "
                f"{self.anchor_frame_ = }, "
                f"{self.previous_frame_ = }, "
                f"{self.motions_ = }"
            )

    def _compute_shift(self, curr_frame: xr.DataArray) -> Shift:
        """
        The simplest way to stabilize streaming frames would be to have a single reference frame
        (the first frame) and shift all subsequent frames against this reference frame.
        However, as different sets of neurons are active at different times, frames that are far
        apart in time sometimes have few common objects to lock onto.

        To mitigate this, we could stabilize all frames against the previous frame, domino style.
        However, errors stack up and a gradual shift takes place with this strategy.

        This algorithm attempts to solve this issue by mixing the two strategies:
        1. We default to stabilizing against the anchor frame.
        2. If we begin losing features to lock onto, the anchor shift will explode to
        an unpredictable value.
        3. In this case, we fall back to the sequential shift.
        4. During the "anchor mismatch period", the sequential shift will slowly drift.
        5. And then, as old features surface again, the anchor will lock in again.
        6. However, the sequential shift will have drifted.
        7. We try to estimate how fast it would drift with drift_speed.
        8. Then, the TRUE shift is within the range of sequential_shift +- drift.
        9. Thus, we assume that if anchor_shift falls within this range, the anchor shift
        has returned to the TRUE shift.

        in mathematical notations, this translates to:
        if:
        sequential_shift - drift_speed < anchor_shift < sequential_shift + drift_speed

        then:
        true_shift = anchor_shift

        the inequality is same as:
        sequential_shift - anchor_shift < drift_speed
        anchor_shift - sequential_shift < drift_speed

        which summarizes to:
        if: abs(sequential_shift - anchor_shift) < drift_speed
        then: true_shift = anchor_shift
        """
        curr = butterworth(curr_frame, **self.filter_kwargs)
        prev = butterworth(self.previous_frame_, **self.filter_kwargs)

        anchor_shift, _, _ = phase_cross_correlation(self.anchor_frame_, curr, **self.pcc_kwargs)
        sequent_shift, _, _ = phase_cross_correlation(prev, curr, **self.pcc_kwargs)

        shift_diff = np.linalg.norm(anchor_shift - sequent_shift)

        frame_idx = curr_frame[AXIS.frame_coord].item()
        drift_threshold = (frame_idx - self._anchor_last_applied_on) * self.drift_speed

        if shift_diff > drift_threshold:
            shift = sequent_shift
        else:
            shift = anchor_shift
            self._anchor_last_applied_on = frame_idx

        return Shift(height=shift[0], width=shift[1])

    def _apply_shift(self, frame: xr.DataArray, shift: Shift) -> xr.DataArray:
        # Define the affine transformation matrix for translation
        M = np.float32([[1, 0, shift.width], [0, 1, shift.height]])

        shifted_frame = cv2.warpAffine(
            frame.values,
            M,
            (frame.sizes[AXIS.width_dim], frame.sizes[AXIS.height_dim]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return xr.DataArray(shifted_frame, dims=frame.dims, coords=frame.coords)

    def _update_anchor(self, frame: xr.DataArray) -> xr.DataArray:
        curr_index = frame[AXIS.frame_coord].item()

        return (self.anchor_frame_ * curr_index + butterworth(frame, **self.filter_kwargs)) / (
            curr_index + 1
        )
