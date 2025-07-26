from glob import glob
from pathlib import Path

import cv2
import numpy as np
import xarray as xr
from skimage import io

from cala.nodes.io import stream


def generate_text_image(
    text: str,
    frame_dims: tuple[int, int] = (256, 256),
    org: tuple[int, int] = (50, 50),
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: int = 1,
) -> np.ndarray:
    image = np.zeros(frame_dims, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    return cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def save_tiff(filename: Path, frame: np.ndarray) -> None:
    io.imsave(filename, frame)


def save_movie(filename: Path, frames: xr.DataArray) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(filename),
        fourcc,
        24.0,
        (frames.sizes["x"], frames.sizes["y"]),
    )

    for frame in frames:
        # grayscale, so convert to BGR color:
        frame_8bit = cv2.cvtColor(frame.astype(np.uint8).values, cv2.COLOR_GRAY2BGR)
        out.write(frame_8bit)

    out.release()


def test_tiff_stream(tmp_path):
    for i in range(10):
        image = generate_text_image(str(i))
        save_tiff(tmp_path / f"{i}.tif", image)

    media = sorted(glob(str(tmp_path / "*.tif")))
    s = iter(stream(media))

    for idx, res in enumerate(s):
        np.testing.assert_array_equal(res, generate_text_image(str(idx)))


def test_video_stream(tmp_path):
    frames = []
    for i in range(10):
        frames.append(xr.DataArray(generate_text_image(str(i)), dims=["x", "y"]))

    video = xr.concat(frames, dim="t")
    save_movie(tmp_path / "video.mp4", video)

    media = sorted(glob(str(tmp_path / "*.mp4")))
    s = iter(stream(media))

    for idx, res in enumerate(s):
        np.testing.assert_allclose(
            res.astype(np.int16), generate_text_image(str(idx)).astype(np.int16), atol=32
        )
