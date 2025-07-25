from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.gui.plots import Plotter
from cala.models.entity import Component
from cala.nodes.iter.residuals import Residuals


def test_init(sample_data: dict[str, Any], plotter: Plotter) -> None:
    """Test the correctness of the residual computation."""
    # Prepare data
    sample_movie = sample_data["movie"]
    sample_denoised = sample_data["denoised"]
    sample_footprints = sample_data["footprints"]
    sample_traces = sample_data["traces"]
    sample_residual = sample_data["residual"]

    isitclean = sample_movie - sample_denoised - sample_residual

    plotter.plot_footprints(sample_footprints, subdir="init/resid")
    plotter.plot_traces(sample_traces, subdir="init/resid")
    plotter.save_video_frames(
        [
            (sample_movie, "movie"),
            (sample_denoised, "denoised"),
            (sample_residual, "residual"),
            (isitclean, "isitclean"),
        ],
        subdir="init/resid",
    )

    initializer = ResidualInitializer(ResidualInitializerParams(buffer_length=len(sample_movie)))

    # Run computation
    initializer.learn_one(sample_footprints, sample_traces, sample_movie)
    result = initializer.transform_one()

    assert np.array_equal(
        sample_residual.transpose("frame", "height", "width"),
        result.transpose("frame", "height", "width"),
    )


def test_update() -> None: ...
