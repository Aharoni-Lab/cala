import numpy as np
import pytest
import xarray as xr

from cala.nodes.iter.pixel_stats import PixelStater


def test_init() -> None:
    """Test the correctness of the pixel statistics computation."""

    # Run computation
    initializer.learn_one(traces, video)
    result = initializer.transform_one().transpose("component", "width", "height")

    label = (video @ traces).transpose("component", "width", "height") / video.sizes["frame"]

    assert np.array_equal(result, label)


def test_ingest_frame() -> None: ...


def test_ingest_component() -> None: ...
