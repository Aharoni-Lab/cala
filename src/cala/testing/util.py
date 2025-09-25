import cv2
import numpy as np
import xarray as xr


def assert_scalar_multiple_arrays(a: xr.DataArray, b: xr.DataArray, /, rtol: float = 1e-5) -> None:
    """Using the Pythagorean Theorem"""

    if not 0 <= rtol <= 1:
        raise ValueError(f"rtol must be between 0 and 1, got {rtol}.")

    abab = (a @ b) ** 2
    aabb = a.dot(a) * b.dot(b)

    assert abab > aabb * (1 - rtol)


def generate_text_image(
    text: str,
    frame_dims: tuple[int, int] = (256, 256),
    org: tuple[int, int] = None,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: int = 1,
) -> np.ndarray:
    image = np.zeros(frame_dims, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if org is None:
        org = (frame_dims[0] // 2, frame_dims[1] // 2)

    return cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def shift_by(img: np.ndarray, right_pix: int, down_pix: int) -> np.ndarray:
    M = np.float32([[1, 0, right_pix], [0, 1, down_pix]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
