from logging import Logger
from typing import Annotated as A

import cv2
import numpy as np
from noob import Name, process_method
from pydantic import BaseModel, Field, ConfigDict

from cala.assets import Frame
from cala.logging import init_logger
from cala.models import AXIS
from cala.util import package_frame


class Shift(BaseModel):
    x: float  # width
    y: float  # height
    a: float  # angle


class RigidStabilizer(BaseModel):
    drift_speed: float = 1.0
    kwargs: dict = Field(default_factory=dict)

    _anchor_last_applied_on: int = None
    anchor_frame_: np.ndarray = None
    previous_frame_: np.ndarray = None
    previous_keypoints_: np.ndarray = None
    motions_: list[Shift] = Field(default_factory=list)

    clahe: cv2.CLAHE = Field(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)), exclude=True)
    logger: Logger = init_logger(__name__)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @process_method
    def stabilize(self, frame: Frame) -> A[Frame, Name("frame")]:
        arr = frame.array.values.astype(np.uint8)
        farr_opt = self.clahe.apply(arr)

        if self.previous_keypoints_ is None:
            self.previous_keypoints_ = self._find_keypoints(farr_opt)

        else:
            shift = self._generate_shift(farr_opt)
            self.motions_.append(shift)
            self.previous_keypoints_ = self._find_keypoints(farr_opt)
            arr = self._apply_shift(arr, shift)

        self.previous_frame_ = farr_opt
        frame = package_frame(arr, index=frame.array[AXIS.frame_coord].item())

        return frame

    @staticmethod
    def _find_keypoints(frame: np.ndarray) -> np.ndarray:
        """calculate and save GFTT keypoints for current frame"""
        return cv2.goodFeaturesToTrack(
            frame,
            maxCorners=200,
            qualityLevel=0.05,
            minDistance=30.0,
            blockSize=3,
            mask=None,
            useHarrisDetector=False,
            k=0.04,
        )

    def _generate_shift(self, frame: np.ndarray) -> Shift:
        # calculate optical flow using Lucas-Kanade differential method
        curr_kps, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame_, frame, self.previous_keypoints_, None
        )

        # select only valid keypoints
        valid_curr_kps = curr_kps[status == 1]  # current
        valid_prev_kps = self.previous_keypoints_[status == 1]  # previous

        # calculate optimal affine transformation between previous_2_current keypoints
        shift = cv2.estimateAffinePartial2D(valid_prev_kps, valid_curr_kps)[0]

        if shift is not None:
            # translation in x direction
            dx = shift[0, 2]
            # translation in y direction
            dy = shift[1, 2]
            # rotation
            da = np.arctan2(shift[1, 0], shift[0, 0])
        else:
            dx = dy = da = 0

        return Shift(x=dx, y=dy, a=da)

    def _apply_shift(self, frame: np.ndarray, shift: Shift) -> np.ndarray:
        """
        An internal method that applies affine transformation to the given frame
        from previously calculated transformations
        """
        # building 2x3 transformation matrix from extracted transformations
        shift_op = np.zeros((2, 3), np.float32)
        shift_op[0, 0] = np.cos(shift.a)
        shift_op[0, 1] = -np.sin(shift.a)
        shift_op[1, 0] = np.sin(shift.a)
        shift_op[1, 1] = np.cos(shift.a)
        shift_op[0, 2] = shift.x
        shift_op[1, 2] = shift.y

        # Applying an affine transformation to the frame
        return cv2.warpAffine(frame, shift_op, frame.shape[::-1])

    # def update_anchor(self, frame: xr.DataArray) -> xr.DataArray:
    #     curr_index = frame[AXIS.frame_coord].item()
    #
    #     return (self.anchor_frame_.array * curr_index + frame) / (curr_index + 1)

    #     if self.is_first_frame(frame):
    #         return frame
    #
    #     curr_frame = frame.array
    #
    #     shift = self.compute_shift(curr_frame)
    #     shifted_frame = self.apply_shift(curr_frame, shift)
    #
    #     self.previous_frame_ = Frame.from_array(shifted_frame)
    #
    #     if self._anchor_last_applied_on == shifted_frame[AXIS.frame_coord].item():
    #         self.anchor_frame_.array = self.update_anchor(shifted_frame)
    #
    #     self.motions_.append(shift)
    #
    #     return Frame.from_array(
    #         xr.DataArray(shifted_frame, dims=frame.array.dims, coords=frame.array.coords)
    #     )

    # def is_first_frame(self, frame: Frame) -> bool:
    #     if (
    #         (self.anchor_frame_ is not None)
    #         and (self.previous_frame_ is not None)
    #         and (self.motions_ is not None)
    #     ):
    #         return False
    #
    #     elif (
    #         (self.anchor_frame_ is None)
    #         and (self.previous_frame_ is None)
    #         and (self.motions_ is None)
    #     ):
    #         self._anchor_last_applied_on = 0
    #         self.anchor_frame_ = frame
    #         self.previous_frame_ = frame
    #         self.motions_ = [Shift(width=0, height=0)]
    #         return True
    #
    #     else:
    #         raise NotImplementedError(
    #             f"Undefined State: Only some of the attributes are initialized: "
    #             f"{self.anchor_frame_ = }, "
    #             f"{self.previous_frame_ = }, "
    #             f"{self.motions_ = }"
    #         )

    # def compute_shift(self, curr_frame: xr.DataArray) -> Shift:
    #     """
    #     The simplest way to stabilize streaming frames would be to have a single reference frame
    #     (the first frame) and shift all subsequent frames against this reference frame.
    #     However, as different sets of neurons are active at different times, frames that are far
    #     apart in time sometimes have few common objects to lock onto.
    #
    #     To mitigate this, we could stabilize all frames against the previous frame, domino style.
    #     However, errors stack up and a gradual shift takes place with this strategy.
    #
    #     This algorithm attempts to solve this issue by mixing the two strategies:
    #     1. We default to stabilizing against the anchor frame.
    #     2. If we begin losing features to lock onto, the anchor shift will explode to
    #     an unpredictable value.
    #     3. In this case, we fall back to the sequential shift.
    #     4. During the "anchor mismatch period", the sequential shift will slowly drift.
    #     5. And then, as old features surface again, the anchor will lock in again.
    #     6. However, the sequential shift will have drifted.
    #     7. We try to estimate how fast it would drift with drift_speed.
    #     8. Then, the TRUE shift is within the range of sequential_shift +- drift.
    #     9. Thus, we assume that if anchor_shift falls within this range, the anchor shift
    #     has returned to the TRUE shift.
    #
    #     in mathematical notations, this translates to:
    #     if:
    #     sequential_shift - drift_speed < anchor_shift < sequential_shift + drift_speed
    #
    #     then:
    #     true_shift = anchor_shift
    #
    #     the inequality is same as:
    #     sequential_shift - anchor_shift < drift_speed
    #     anchor_shift - sequential_shift < drift_speed
    #
    #     which summarizes to:
    #     if: abs(sequential_shift - anchor_shift) < drift_speed
    #     then: true_shift = anchor_shift
    #     """
    #
    #     anchor_shift, a_error, _ = phase_cross_correlation(
    #         self.anchor_frame_.array, curr_frame.values, **self.kwargs
    #     )
    #
    #     sequent_shift, s_error, _ = phase_cross_correlation(
    #         self.previous_frame_.array, curr_frame.values, **self.kwargs
    #     )
    #
    #     shift_diff = abs(np.linalg.norm(anchor_shift - sequent_shift))
    #
    #     frame_idx = curr_frame[AXIS.frame_coord].item()
    #     drift_threshold = (frame_idx - self._anchor_last_applied_on) * self.drift_speed
    #
    #     if shift_diff > drift_threshold:
    #         shift = sequent_shift
    #     else:
    #         shift = anchor_shift
    #         self._anchor_last_applied_on = frame_idx
    #
    #     return Shift(height=shift[0], width=shift[1])

    # def apply_shift(self, frame: xr.DataArray, shift: Shift) -> xr.DataArray:
    #     # Define the affine transformation matrix for translation
    #     M = np.float32([[1, 0, shift.width], [0, 1, shift.height]])
    #
    #     shifted_frame = cv2.warpAffine(
    #         frame.values,
    #         M,
    #         (frame.sizes[AXIS.width_dim], frame.sizes[AXIS.height_dim]),
    #         flags=cv2.INTER_LINEAR,
    #         borderMode=cv2.BORDER_CONSTANT,
    #         borderValue=np.nan,
    #     )
    #     shifted_frame = np.nan_to_num(shifted_frame, copy=True, nan=0)
    #     return xr.DataArray(shifted_frame, dims=frame.dims, coords=frame.coords)
