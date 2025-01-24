import pytest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from cala.segmentation.detect.base import BaseDetector
from cala.segmentation.detect import MaxProjection, LaplacianOfGaussian


@pytest.fixture(
    params=[
        (
            MaxProjection,
            {
                "core_axes": ["height", "width"],
                "iter_axis": "frames",
                "local_max_radius": 8,
                "intensity_threshold": 1,
            },
        ),
        (
            LaplacianOfGaussian,
            {
                "core_axes": ["height", "width"],
                "iter_axis": "frames",
            },
        ),
    ],
    ids=["max_proj", "log"],
)
def detector_instance(request):
    """Parameterized fixture that yields instances of all detector classes.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request object containing the parameter tuple (class, params)

    Returns
    -------
    BaseDetector
        An instance of a detector class initialized with specified parameters

    Notes
    -----
    Each parameter is a tuple of (DetectorClass, param_dict) where:
    - DetectorClass is the class to instantiate
    - param_dict contains the initialization parameters for that class
    """
    detector_class, params = request.param
    return detector_class(**params)


def visualize_detection(
    video: xr.DataArray,
    detector: BaseDetector,
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

    # Plot ground truth
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

    # Plot detected seeds
    if isinstance(detector, MaxProjection):
        ax.scatter(
            seeds["width"],
            seeds["height"],
            color="r",
            marker="x",
            s=100,
            label="Detected",
            alpha=0.7,
        )
    elif isinstance(detector, LaplacianOfGaussian):
        # Only add label to first circle for single legend entry
        for idx, blob in seeds.iterrows():
            proj_idx, y, x, r = blob
            c = plt.Circle(
                (x, y),
                r,
                color="r",
                linewidth=2,
                fill=False,
                label="Detected" if idx == 0 else None,
                alpha=0.7,
            )
            ax.add_patch(c)
        ax.set_axis_off()

    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

    return fig
