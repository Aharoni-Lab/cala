from typing import Annotated as A

import xarray as xr
from noob import Name
from noob.node import Node
from pydantic import ConfigDict
from skimage.restoration import estimate_sigma
from sklearn.feature_extraction.image import PatchExtractor

from cala.assets import Frame, Residual
from cala.models import AXIS


class Energy(Node):
    gaussian_std: float
    """not sure why we're smoothing the residual...??"""
    min_frames: int
    """minimum number of frames to consider to begin detecting cells"""

    noise_level_: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def process(
        self, residuals: Residual, trigger: Frame
    ) -> A[xr.DataArray | None, Name("energy")]:
        if residuals.array is None or residuals.array.sizes[AXIS.frames_dim] < self.min_frames:
            return xr.DataArray()

        residuals = residuals.array
        self.noise_level_ = self._estimate_gaussian_noise(residuals)

        V = self._center_to_median(residuals)

        if (V.max() - V.min()) / 2 <= self.noise_level_:  # if fluctuation is noise level
            return None

        # Compute energy (variance)  -- why are we giving real value to below median? floor it?
        E = (V**2).sum(dim=AXIS.frames_dim)

        return E

    def _estimate_gaussian_noise(self, residuals: xr.DataArray) -> float:
        sampler = PatchExtractor(patch_size=(20, 20), max_patches=30)
        patches = sampler.transform(residuals)
        return float(estimate_sigma(patches))

    def _center_to_median(self, arr: xr.DataArray) -> xr.DataArray:
        """Process residuals through median subtraction and spatial filtering."""
        # Center residuals: why median and not mean?
        pixels_median = arr.median(dim=AXIS.frames_dim)
        V = arr - pixels_median

        return V
