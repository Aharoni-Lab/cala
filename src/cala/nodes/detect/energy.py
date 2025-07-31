import xarray as xr
from noob.node import Node
from pydantic import ConfigDict
from scipy.ndimage import gaussian_filter
from skimage.restoration import estimate_sigma
from sklearn.feature_extraction.image import PatchExtractor

from cala.assets import Residual
from cala.models import AXIS


class Energy(Node):
    gaussian_std: float

    noise_level_: float = None
    sampler: PatchExtractor = PatchExtractor(patch_size=(20, 20), max_patches=30)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def process(self, residuals: Residual) -> xr.DataArray | None:
        residuals = residuals.array
        frame_shape = residuals[0].shape
        self.noise_level_ = self._estimate_gaussian_noise(residuals, frame_shape)

        V = self._center_to_median(residuals)

        if (V.max() - V.min()) / 2 <= self.noise_level_:  # if fluctuation is noise level
            return None

        # Compute energy (variance)  -- why are we giving real value to below median? floor it?
        E = (V**2).sum(dim=AXIS.frames_dim)

        return E

    def _estimate_gaussian_noise(
        self, residuals: xr.DataArray, frame_shape: tuple[int, ...]
    ) -> float:
        self.sampler.patch_size = min(self.sampler.patch_size, frame_shape)
        patches = self.sampler.transform(residuals)
        return float(estimate_sigma(patches))

    def _center_to_median(self, arr: xr.DataArray) -> xr.DataArray:
        """Process residuals through median subtraction and spatial filtering."""
        # Center residuals: why median and not mean?
        pixels_median = arr.median(dim=AXIS.frames_dim)
        arr_centered = arr - pixels_median

        # Apply spatial filter -- why are we doing this??
        V = xr.apply_ufunc(
            lambda x: gaussian_filter(x, self.gaussian_std),
            arr_centered,
            input_core_dims=[[*AXIS.spatial_dims]],
            output_core_dims=[[*AXIS.spatial_dims]],
            vectorize=True,
            dask="allowed",
        )

        return V
