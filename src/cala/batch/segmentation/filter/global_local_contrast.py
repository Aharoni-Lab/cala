from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from xarray import DataArray

from .base import BaseFilter


@dataclass
class GLContrastFilter(BaseFilter):
    """Filter seeds based on global-to-local contrast ratio analysis.

    This filter analyzes the relationship between global and local signal variations
    to identify valid seeds. Seeds are considered valid if they show a high ratio
    of global to local RMS (Root Mean Square), indicating regions where the signal
    has distinct temporal patterns compared to local noise.

    The filter assumes that valid cells exhibit:
    - High global RMS indicating significant overall activity
    - Low local RMS indicating smooth, non-noisy temporal patterns
    - High ratio between global and local RMS

    Notes
    -----
    The filtering process follows these steps:
    1. Computes global RMS over the entire signal
    2. Computes local RMS using rolling windows
    3. Calculates ratio of global to local RMS
    4. Marks seeds as valid where ratio exceeds threshold

    The global-to-local contrast ratio helps identify signal-like behavior
    by comparing overall signal strength to local noise levels.
    """

    window_size: int = 50
    """Number of samples in the rolling window for local RMS calculation."""
    ratio_threshold: float = 1.75
    """Threshold for deciding 'signal-like' behavior."""
    _stateless = True
    """Whether the filter is stateless. Always True for this filter."""

    def fit_kernel(self, X: DataArray, seeds: DataFrame) -> None:
        """Empty implementation for stateless filter."""
        pass

    def transform_kernel(self, X: DataArray, seeds: DataFrame) -> DataFrame:
        """Transform seeds by filtering based on global-to-local contrast ratio.

        Parameters
        ----------
        X : xarray.DataArray
            Input data array containing fluorescence values with dimensions matching
            core_axes and iter_axis.
        seeds : pandas.DataFrame
            DataFrame containing seed coordinates in columns matching core_axes.

        Returns
        -------
        pandas.DataFrame
            Seeds DataFrame with an additional boolean column 'mask_gl' indicating valid seeds.
            Seeds are marked as valid if their global-to-local RMS ratio exceeds
            the ratio_threshold.

        Notes
        -----
        This method:
        1. Extracts fluorescence traces for each seed
        2. Computes global and local RMS values
        3. Calculates their ratio
        4. Thresholds the ratio to identify valid seeds
        """
        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds") for axis in self.core_axes
        }

        # Select the relevant subset from X using vectorized selection
        seed_pixels = X.sel(**seed_das)

        contrast_ratio = xr.apply_ufunc(
            self.rms_kernel,
            seed_pixels,
            input_core_dims=[[self.iter_axis]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Threshold the ratio to produce a boolean mask
        # If ratio > ratio_threshold => we consider it 'signal-like'
        contrast_mask = contrast_ratio > self.ratio_threshold
        seeds["mask_gl"] = contrast_mask.compute().values

        return seeds

    def fit_transform_shared_preprocessing(self, X: DataArray, seeds: DataFrame) -> None:
        """Empty implementation for stateless filter."""
        pass

    def rms_kernel(self, signal: np.ndarray) -> float:
        """Calculate the ratio of global to local RMS for a signal.

        This method computes the ratio between the global Root Mean Square (RMS)
        of the entire signal and the local RMS computed using rolling windows.
        A high ratio indicates regions where the signal has significant overall
        variation but is locally smooth.

        Parameters
        ----------
        signal : numpy.ndarray
            1D array containing the signal to analyze

        Returns
        -------
        float
            Mean ratio of global RMS to local RMS across the signal.
            Higher values indicate stronger signal-like behavior.

        Notes
        -----
        The method:
        1. Computes global RMS over the entire signal
        2. Computes local RMS using rolling standard deviation
        3. Handles edge effects in local RMS calculation
        4. Returns mean ratio of global to local RMS

        The ratio is high when:
        - The signal has significant overall variation (high global RMS)
        - The signal is locally smooth (low local RMS)
        """
        # Global RMS over the entire signal
        global_rms = np.sqrt(np.mean(signal**2))

        # Local RMS: approximate by rolling standard deviation
        local_rms = pd.Series(signal).rolling(self.window_size, center=True).std().to_numpy()

        # Handle edge effects from incomplete windows
        nan_mask = np.isnan(local_rms)
        local_rms[nan_mask] = np.nanmean(local_rms[~nan_mask]) if not all(nan_mask) else 1e-8

        # Compute ratio: global_RMS / local_RMS
        # High ratio => local region is relatively smooth compared to global variation
        ratio = np.divide(
            global_rms, local_rms, out=np.zeros_like(local_rms), where=(local_rms != 0)
        )

        return float(np.mean(ratio))
