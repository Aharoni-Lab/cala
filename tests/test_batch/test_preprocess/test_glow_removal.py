from typing import Any

import numpy as np
import xarray as xr

from cala.batch.preprocess.glow_removal import GlowRemover


def test_glow_remover_fit_transform(raw_calcium_video: xr.DataArray, params: Any) -> None:
    """Test GlowRemover's fit and transform functionality using synthetic data."""
    video = raw_calcium_video

    remover = GlowRemover()
    remover.fit(video)

    # The base_brightness should be approximately the baseline minus some noise
    assert np.abs(remover.base_brightness_.mean() - params.baseline) < params.noise_level * 3

    transformed = remover.transform(video)

    assert isinstance(transformed, xr.DataArray)
    assert transformed.shape == video.shape
    # Check that the minimum values are now close to zero
    assert transformed.min() > -params.noise_level * 3
    assert transformed.min() < params.noise_level * 3
