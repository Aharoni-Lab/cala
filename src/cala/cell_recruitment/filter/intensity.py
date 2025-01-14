from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseFilter


@dataclass
class IntensityFilter(BaseFilter):
    seed_intensity_factor: int = 2
    max_brightness_projection_: xr.DataArray = None
    intensity_threshold_: float = field(default=None)

    def fit_kernel(self, X: xr.DataArray, y=None):
        num_projection_pixels = np.prod(
            [self.max_brightness_projection_.sizes[axis] for axis in self.core_axes]
        )

        bins = max(1, int(round(num_projection_pixels / 10)))
        hist, edges = np.histogram(self.max_brightness_projection_.values, bins=bins)

        # Determine the peak of the histogram
        peak_idx = np.argmax(hist)
        scaled_peak_idx = int(round(peak_idx * self.seed_intensity_factor))

        self.intensity_threshold_ = edges[scaled_peak_idx]

        return self

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame):
        # Create the mask based on the stored threshold
        mask = (self.max_brightness_projection_ > self.intensity_threshold_).stack(
            {self.spatial_axis: self.core_axes}
        )
        mask_df = mask.to_pandas().rename("mask_int").reset_index()

        # Merge the mask with seeds
        filtered_seeds = pd.merge(seeds, mask_df, on=self.core_axes, how="left")

        return filtered_seeds

    def fit_transform_shared_preprocessing(self, X: xr.DataArray, y=None):
        self.max_brightness_projection_ = X.max(self.iter_axis)
