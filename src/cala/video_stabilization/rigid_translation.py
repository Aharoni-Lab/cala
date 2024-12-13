from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .base import BaseMotionCorrector


@dataclass
class RigidTranslator(BaseMotionCorrector):
    anchor_frame_index: Optional[int] = None

    def _fit_kernel(self, current_frame: np.ndarray, **kwargs) -> Dict[str, float]:
        if self.anchor_frame_ is None:
            raise ValueError("Base frame has not been established.")

        shift, error, diffphase = phase_cross_correlation(
            self.anchor_frame_,
            current_frame,
            upsample_factor=kwargs.get("upsample_factor", 1),
        )
        return shift  # Returns an array [shift_y, shift_x] --> self.motion_

    def _transform_kernel(self, frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
        # Define the affine transformation matrix for translation
        M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])

        transformed_frame = cv2.warpAffine(
            frame,
            M,
            (frame.shape[1], frame.shape[0]),  # (Width, Height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        return transformed_frame
