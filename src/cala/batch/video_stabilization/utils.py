import cv2
import numpy as np


def check_temp(fm: np.ndarray, max_sh: int) -> float:
    """
    Compute the circularity metric for a frame.
    """
    fm_pad = np.pad(fm, max_sh)
    cor = cv2.matchTemplate(
        fm.astype(np.float32),
        fm_pad.astype(np.float32),
        cv2.TM_SQDIFF_NORMED,
    )
    contours, _ = cv2.findContours(
        (cor < 1).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) != 1:
        return 0.0
    contour = contours[0]
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter**2))
    return circularity
