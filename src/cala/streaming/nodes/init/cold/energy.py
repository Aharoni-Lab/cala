from dataclasses import dataclass, field

import xarray as xr
from scipy.ndimage import gaussian_filter
from skimage.restoration import estimate_sigma
from sklearn.feature_extraction.image import PatchExtractor

from cala.streaming.core import Axis, Parameters
from cala.streaming.nodes import Node
from cala.streaming.stores.odl import Residuals


@dataclass
class EnergyParams(Parameters, Axis):
    gaussian_std: float

    def validate(self) -> None: ...


@dataclass
class Energy(Node):
    params: EnergyParams
    noise_level_: float = field(init=False)
    sampler: PatchExtractor = PatchExtractor(patch_size=(20, 20), max_patches=30)

    def process(self, residuals: Residuals) -> xr.DataArray | None:
        frame_shape = residuals[0].shape
        self.noise_level_ = self._estimate_gaussian_noise(residuals, frame_shape)

        V = self._center_to_median(residuals)

        if (V.max() - V.min()) / 2 <= self.noise_level_:  # if fluctuation is noise level
            return None

        # Compute energy (variance)  -- why are we giving real value to below median? floor it?
        E = (V**2).sum(dim=self.params.frames_dim)

        return E

    def _estimate_gaussian_noise(self, residuals: Residuals, frame_shape: tuple[int, ...]) -> float:
        self.sampler.patch_size = min(self.sampler.patch_size, frame_shape)
        patches = self.sampler.transform(residuals)
        return float(estimate_sigma(patches))

    def _center_to_median(self, arr: xr.DataArray) -> xr.DataArray:
        """Process residuals through median subtraction and spatial filtering."""
        # Center residuals: why median and not mean?
        pixels_median = arr.median(dim=self.params.frames_dim)
        arr_centered = arr - pixels_median

        # Apply spatial filter -- why are we doing this??
        V = xr.apply_ufunc(
            lambda x: gaussian_filter(x, self.params.gaussian_std),
            arr_centered,
            input_core_dims=[[*self.params.spatial_dims]],
            output_core_dims=[[*self.params.spatial_dims]],
            vectorize=True,
            dask="allowed",
        )

        return V
