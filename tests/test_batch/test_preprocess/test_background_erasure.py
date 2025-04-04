import numpy as np
import pytest
import xarray as xr

from cala.batch.preprocess.background_erasure import BackgroundEraser


def test_background_erasure_uniform(raw_calcium_video: xr.DataArray) -> None:
    video = raw_calcium_video

    # Test uniform method
    eraser = BackgroundEraser(core_axes=["height", "width"], method="uniform", kernel_size=5)

    result = eraser.fit_transform(video)

    assert result.shape == video.shape
    assert result.dims == video.dims
    assert not np.allclose(result, video)  # Should be different from input
    assert result.name == f"{video.name}_subtracted"

    # Not sure how exactly to test. Largest feature is some size, but how do you measure "size" here.
    assert result.mean() < video.mean()


def test_background_erasure_tophat(raw_calcium_video: xr.DataArray) -> None:
    video = raw_calcium_video

    # Test tophat method
    eraser = BackgroundEraser(core_axes=["height", "width"], method="tophat", kernel_size=5)

    result = eraser.fit_transform(video)

    assert result.shape == video.shape
    assert result.dims == video.dims
    assert not np.allclose(result, video)
    assert result.name == f"{video.name}_subtracted"

    # Not sure how exactly to test. Largest feature is some size, but how do you measure "size" here.
    assert result.mean() < video.mean()


def test_invalid_method(raw_calcium_video: xr.DataArray) -> None:
    video = raw_calcium_video

    with pytest.raises(ValueError):
        eraser = BackgroundEraser(method="invalid")
        eraser.fit_transform(video)
