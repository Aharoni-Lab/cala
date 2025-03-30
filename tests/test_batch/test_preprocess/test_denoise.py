import numpy as np
import pytest

from cala.batch.preprocess.denoise import Denoiser


@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
def test_denoiser_methods(raw_calcium_video, method):
    video = raw_calcium_video

    if method == "median":
        kwargs = {"ksize": 3}
    elif method == "bilateral":
        kwargs = {"d": 3, "sigmaColor": 75, "sigmaSpace": 75}
    elif method == "gaussian":
        kwargs = {"ksize": [3, 3], "sigmaX": 1.5}

    denoiser = Denoiser(method=method, kwargs=kwargs)
    result = denoiser.fit_transform(video.astype(np.float32))

    # Check basic properties
    assert result.shape == video.shape
    assert isinstance(result, type(video))  # Should still be an xarray
    assert not np.allclose(result, video)  # Should be different from input
    if not video.name:
        assert result.name == "denoised"
    else:
        assert result.name == f"{video.name}_denoised"
    # Denoising should reduce variance while preserving mean
    assert result.var() < video.var()
    assert np.abs(result.mean() - video.mean()) < 0.1 * video.mean()


def test_invalid_method(raw_calcium_video):
    video = raw_calcium_video

    with pytest.raises(ValueError, match="denoise method 'invalid' not understood"):
        denoiser = Denoiser(method="invalid")
        denoiser.fit_transform(video)
