from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from .base import BaseFilter


@dataclass
class GMMFilter(BaseFilter):
    quantile_floor: float = 0.1
    quantile_ceil: float = 99.9
    num_components: int = 2
    num_valid_components: int = 1
    mean_mask: bool = True
    gmm_: GaussianMixture = None
    valid_component_indices_: np.ndarray = None

    def __post_init__(self):
        if self.quantile_floor >= self.quantile_ceil:
            raise ValueError("quantile_floor must be smaller than quantile_ceil")

    @property
    def quantiles(self):
        return self.quantile_floor, self.quantile_ceil

    def fit_kernel(self, X: xr.DataArray, y=None) -> None:
        self.gmm_ = GaussianMixture(n_components=self.num_components, random_state=42)
        self.gmm_.fit(X)

        valid_component_indices = np.argsort(self.gmm_.means_.reshape(-1))[
            -self.num_valid_components :
        ]
        self.valid_component_indices_ = valid_component_indices

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
        # Select the spatial points corresponding to the seeds
        spatial_coords = seeds[self.core_axes].apply(tuple, axis=1).tolist()
        seed_pixels = X.sel({self.spatial_axis: spatial_coords})

        # Compute both percentiles in a single quantile call
        quantiles = seed_pixels.quantile(
            q=self.quantiles,
            dim=self.iter_axis,
            interpolation="linear",
        )
        seed_valley = quantiles.sel(q=quantiles[0])
        seed_peak = quantiles.sel(q=quantiles[1])
        seed_amplitude = seed_peak - seed_valley
        seed_amplitudes = seed_amplitude.compute().values.reshape(-1, 1)

        # Predict cluster assignments and determine validity
        cluster_labels = self.gmm_.predict(seed_amplitudes)
        is_valid = np.isin(cluster_labels, self.valid_component_indices_)

        # Apply mean mask if required
        if self.mean_mask:
            lowest_mean = np.min(self.gmm_.means_)
            mean_mask_condition = seed_amplitudes.flatten() > lowest_mean
            is_valid &= mean_mask_condition

        # Update the seeds DataFrame with the mask
        seeds["mask_gmm"] = is_valid

        return seeds

    def fit(self, X: xr.DataArray, y=None) -> Self:
        self.fit_kernel(X)
        return self

    def transform(self, X: xr.DataArray, y=None) -> pd.DataFrame:
        if self.valid_component_indices_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. Please call 'fit' first."
            )

        return self.transform_kernel(X, y)

    def fit_transform_shared_preprocessing(self, X, seeds):
        pass
