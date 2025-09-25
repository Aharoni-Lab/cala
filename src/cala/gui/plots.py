from pathlib import Path

import cv2
import numpy as np
import xarray as xr


def write_movie(video: xr.DataArray, path: str | Path) -> None:
    """Test visualization of stabilized calcium video to verify motion stabilization."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 24.0, (video.sizes["width"], video.sizes["height"]))

    max_brightness = video.max().item()

    for frame in video:
        frame_8bit = (frame / max_brightness * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_8bit.values, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()
