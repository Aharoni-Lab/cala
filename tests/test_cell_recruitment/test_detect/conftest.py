import pytest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def visualize_detection(
    video: xr.DataArray,
    seeds: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None,
    frame_idx: Optional[int] = None,
    title: str = "Detected Seeds",
) -> plt.Figure:
    """Visualize detected seeds overlaid on the video."""
    fig, ax = plt.subplots(figsize=(10, 10))

    if frame_idx is None:
        frame = video.max(dim="frames")
    else:
        frame = video.isel(frames=frame_idx)

    vmin, vmax = np.percentile(frame, [1, 99])
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)

    # Plot detected seeds
    ax.scatter(
        seeds["width"],
        seeds["height"],
        color="r",
        marker="x",
        s=100,
        label="Detected",
        alpha=0.7,
    )

    # Plot ground truth if provided
    if ground_truth is not None:
        ax.scatter(
            ground_truth["width"],
            ground_truth["height"],
            color="g",
            marker="o",
            s=150,
            facecolors="none",
            label="Ground Truth",
            alpha=0.7,
        )

    ax.legend()
    ax.set_title(title)

    return fig
