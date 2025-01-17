from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from .base import BaseFilter


@dataclass
class GMMFilter(BaseFilter):
    quantile_floor: float = 0.1
    quantile_ceil: float = 0.99
    num_components: int = 2
    num_valid_components: int = 1
    mean_mask: bool = True
    seed_amplitude_: np.ndarray = field(init=False)
    gmm_: GaussianMixture = field(init=False)
    valid_component_indices_: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.quantile_floor >= self.quantile_ceil:
            raise ValueError("quantile_floor must be smaller than quantile_ceil")
        if self.quantile_floor < 0 or self.quantile_ceil > 1:
            raise ValueError("quantiles must be between 0 and 1")
        self.gmm_ = GaussianMixture(random_state=42)

    @property
    def quantiles(self):
        return self.quantile_floor, self.quantile_ceil

    def fit_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
        self.gmm_.set_params(n_components=self.num_components)
        self.gmm_.fit(self.seed_amplitude_)

        self.valid_component_indices_ = np.argsort(self.gmm_.means_.reshape(-1))[
            -self.num_valid_components :
        ]  # get the indices of the components with the highest means

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
        # Predict cluster assignments and determine validity
        cluster_labels = self.gmm_.predict(self.seed_amplitude_)
        is_valid = np.isin(cluster_labels, self.valid_component_indices_)

        # Apply mean mask if required
        if self.mean_mask:
            lowest_mean = np.min(self.gmm_.means_)
            mean_mask_condition = self.seed_amplitude_.flatten() > lowest_mean
            is_valid &= mean_mask_condition

        # Update the seeds DataFrame with the mask
        seeds["mask_gmm"] = is_valid

        return seeds

    def fit_transform_shared_preprocessing(self, X: xr.DataArray, seeds):
        # Select the spatial points corresponding to the seeds
        spatial_coords = seeds[self.core_axes].apply(tuple, axis=1).tolist()
        X = X.stack({self.spatial_axis: self.core_axes})
        seed_pixels = X.sel({self.spatial_axis: spatial_coords})

        # Compute both percentiles in a single quantile call
        quantiles = seed_pixels.quantile(
            q=self.quantiles,
            dim=self.iter_axis,
            interpolation="linear",
        )
        seed_valley = quantiles.sel(quantile=self.quantiles[0])
        seed_peak = quantiles.sel(quantile=self.quantiles[1])
        self.seed_amplitude_ = (seed_peak - seed_valley).compute().values.reshape(-1, 1)
