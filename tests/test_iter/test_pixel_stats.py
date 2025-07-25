import numpy as np
import pytest
import xarray as xr

from cala.nodes.iter.pixel_stats import PixelStats


def test_init() -> None:
    """Test the correctness of the pixel statistics computation."""

    # Run computation
    initializer.learn_one(traces, video)
    result = initializer.transform_one().transpose("component", "width", "height")

    label = (video @ traces).transpose("component", "width", "height") / video.sizes["frame"]

    plotter.plot_traces(traces, subdir="init/pixel_stats/sanity_check")
    plotter.plot_pixel_stats(result, subdir="init/pixel_stats/sanity_check")

    assert np.array_equal(result, label)


def test_ingest_frame() -> None: ...


def test_ingest_component() -> None: ...
