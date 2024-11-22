from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseFilter


@dataclass
class IntensityFilter(BaseFilter):
    seed_intensity_factor: int = 2
    fit_transform: bool = True
    max_brightness_projection_: xr.DataArray = None
    intensity_threshold_: float = field(default=None)

    def fit_kernel(self, X):
        self.max_brightness_projection_ = X.max(self.iter_axis)
        num_projection_pixels = np.prod(
            self.max_brightness_projection_.sizes[axis] for axis in self.core_axes
        )

        bins = max(1, int(round(num_projection_pixels / 10)))
        hist, edges = np.histogram(self.max_brightness_projection_.values, bins=bins)

        # Determine the peak of the histogram
        peak_idx = np.argmax(hist)
        scaled_peak_idx = int(round(peak_idx * self.seed_intensity_factor))

        self.intensity_threshold_ = edges[scaled_peak_idx]

        return self

    def transform_kernel(self, X, seeds):
        if self.intensity_threshold_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. Please call 'fit' first."
            )

        if not self.fit_transform:
            self.max_brightness_projection_ = X.max(self.iter_axis)

        # Create the mask based on the stored threshold
        mask = (self.max_brightness_projection_ > self.intensity_threshold_).stack(
            spatial=self.core_axes
        )
        mask_df = mask.to_pandas().rename("mask_int").reset_index()

        # Merge the mask with seeds
        filtered_seeds = pd.merge(seeds, mask_df, on=self.core_axes, how="left")

        return filtered_seeds

    def fit(self, X, y=None):
        self.fit_kernel(X)
        return self

    def transform(self, X, y=None):
        return self.transform_kernel(X, y)
