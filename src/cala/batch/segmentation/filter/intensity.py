from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseFilter


@dataclass
class IntensityFilter(BaseFilter):
    """Filter seeds based on maximum intensity threshold analysis.

    This filter analyzes the maximum intensity of each pixel over time to identify
    valid seeds. Seeds are considered valid if their maximum intensity exceeds
    a threshold determined from the intensity distribution of the entire image.

    The filter assumes that valid cells exhibit:
    - Higher fluorescence intensities compared to background
    - Peak intensities that stand out from the typical intensity distribution

    Parameters
    ----------
    seed_intensity_factor : int, default=2
        Factor to scale the histogram peak index when determining threshold.
        Higher values result in more stringent intensity thresholds.

    Notes
    -----
    The filtering process follows these steps:
    1. Computes maximum intensity projection across time
    2. Creates histogram of pixel intensities
    3. Finds peak of histogram (typical background intensity)
    4. Sets threshold at scaled distance from peak
    5. Marks seeds as valid where intensity exceeds threshold

    The intensity-based approach helps identify bright regions that are
    likely to correspond to active cells rather than background.
    """

    seed_intensity_factor: int = 2
    """Factor to scale the histogram peak for threshold determination."""
    max_brightness_projection_: xr.DataArray = field(init=False)
    """Maximum intensity projection of the input data across time."""
    intensity_threshold_: float = field(init=False)
    """Computed intensity threshold for valid seeds."""

    def fit_kernel(self, X: xr.DataArray, seeds: pd.DataFrame | None = None) -> "IntensityFilter":
        """Learn the intensity threshold from the maximum brightness projection.

        Parameters
        ----------
        X : xarray.DataArray
            Input fluorescence data with dimensions matching core_axes and iter_axis.
        seeds : pandas.DataFrame, optional
            Not used in this filter.

        Returns
        -------
        self : IntensityFilter
            Returns the instance for method chaining.

        Notes
        -----
        This method:
        1. Computes number of pixels for histogram binning
        2. Creates histogram of maximum intensities
        3. Finds peak of histogram (most common intensity)
        4. Sets threshold at scaled distance from peak
        """
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

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
        """Transform seeds by filtering based on maximum intensity.

        Parameters
        ----------
        X : xarray.DataArray
            Input fluorescence data with dimensions matching core_axes and iter_axis.
        seeds : pandas.DataFrame
            DataFrame containing seed coordinates in columns matching core_axes.

        Returns
        -------
        pandas.DataFrame
            Seeds DataFrame with an additional boolean column 'mask_int' indicating valid seeds.
            Seeds are marked as valid if their maximum intensity exceeds the learned threshold.

        Notes
        -----
        This method:
        1. Creates mask from maximum intensity projection using threshold
        2. Stacks spatial dimensions for comparison with seeds
        3. Merges mask with seed coordinates
        """
        # Create the mask based on the stored threshold
        mask = (self.max_brightness_projection_ > self.intensity_threshold_).stack(
            {self.spatial_axis: self.core_axes}
        )
        mask_df = mask.to_pandas().rename("mask_int").reset_index()

        # Merge the mask with seeds
        filtered_seeds = pd.merge(seeds, mask_df, on=self.core_axes, how="left")

        return filtered_seeds

    def fit_transform_shared_preprocessing(
        self, X: xr.DataArray, seeds: pd.DataFrame | None = None
    ) -> None:
        """Compute maximum intensity projection across time.

        Parameters
        ----------
        X : xarray.DataArray
            Input fluorescence data with dimensions matching core_axes and iter_axis.
        seeds : pandas.DataFrame, optional
            Not used in this filter.

        Notes
        -----
        This method:
        1. Computes maximum intensity across the iteration axis (typically time)
        2. Stores result for use in both fitting and transformation
        """
        self.max_brightness_projection_ = X.max(self.iter_axis)
