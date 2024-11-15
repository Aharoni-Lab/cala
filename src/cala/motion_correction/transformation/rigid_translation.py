from typing import Dict

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .base import Transformation


class RigidTranslation(Transformation):
    def compute_shift(
        self, base_frame: np.ndarray, current_frame: np.ndarray
    ) -> Dict[str, float]:
        shift, error, diffphase = phase_cross_correlation(
            base_frame, current_frame, upsample_factor=10
        )
        return {"shift_y": shift[0], "shift_x": shift[1]}

    def apply_transformation(
        self, frame: np.ndarray, params: Dict[str, float]
    ) -> np.ndarray:
        # Define the affine transformation matrix for translation
        M = np.float32([[1, 0, params["shift_x"]], [0, 1, params["shift_y"]]])

        # Apply the affine transformation using warpAffine
        transformed_frame = cv2.warpAffine(
            frame,
            M,
            (frame.shape[1], frame.shape[0]),  # (Width, Height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,  # Fill empty areas with nan
        )
        return transformed_frame
