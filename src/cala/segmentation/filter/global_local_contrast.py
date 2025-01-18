from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from xarray import DataArray

from .base import BaseFilter


@dataclass
class GLContrastFilter(BaseFilter):
    window_size: int = 50
    ratio_threshold: float = 1.75
    """
    window_size : int
        Number of samples in the rolling window for local RMS calculation.
    ratio_threshold : float
        Threshold for deciding "signal-like" behavior.
        Larger ratio => local RMS is relatively small vs. global RMS.
    """

    def fit_kernel(self, X: DataArray, seeds: DataFrame):
        pass

    def transform_kernel(self, X: DataArray, seeds: DataFrame):
        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: Dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds")
            for axis in self.core_axes
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

        # 4) Threshold the ratio to produce a boolean mask
        #    If ratio > ratio_threshold => we consider it 'signal-like'
        contrast_mask = contrast_ratio > self.ratio_threshold
        seeds["mask_dist"] = contrast_mask.compute().values

        return seeds

    def fit_transform_shared_preprocessing(self, X: DataArray, seeds: DataFrame):
        pass

    def rms_kernel(self, signal: np.ndarray) -> float:
        # 1) Global RMS over the entire signal
        global_rms = np.sqrt(np.mean(signal**2))

        # 2) Local RMS: we can approximate by a rolling standard deviation
        local_rms = (
            pd.Series(signal).rolling(self.window_size, center=True).std().to_numpy()
        )

        # Some edges may be NaN because of incomplete windows
        # We replace them with a small value (e.g., np.nanmean) or zero
        # so we don't get spurious large ratios near boundaries:
        nan_mask = np.isnan(local_rms)
        local_rms[nan_mask] = (
            np.nanmean(local_rms[~nan_mask]) if not all(nan_mask) else 1e-8
        )

        # 3) Compute ratio: global_RMS / local_RMS
        #    High ratio => local region is relatively smooth compared to global variation.
        ratio = np.divide(
            global_rms, local_rms, out=np.zeros_like(local_rms), where=(local_rms != 0)
        )

        return np.mean(ratio)
