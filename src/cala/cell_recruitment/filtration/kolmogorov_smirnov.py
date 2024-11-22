from dataclasses import dataclass
import dask as da
from typing import Dict
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kstest, zscore
from .base import BaseFilter


@dataclass
class KSFilter(BaseFilter):
    """
    Filter the seeds using Kolmogorov-Smirnov (KS) test.

    This function assume that the valid seeds’ fluorescence across frames
    notionally follows a bimodal distribution: with a large normal distribution
    representing baseline activity, and a second peak representing when the
    seed/cell is active. KS allows to discard the seeds where the
    null-hypothesis (i.e. the fluorescence intensity is simply a normal
    distribution) is rejected at `sig`.
    sig : float, optional
        The significance threshold to reject null-hypothesis. By default `0.01`.
    """

    significance_threshold: float = 0.05

    def fit_kernel(self, X: xr.DataArray = None):
        pass

    def fit(self, X: xr.DataArray = None, y: pd.DataFrame = None):
        return self

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame):
        """
        Returns : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_ks",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
        """
        if not isinstance(X.data, da.array.core.Array):
            X = X.chunk(auto=True)  # Let Dask decide chunk sizes

        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: Dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds")
            for axis in self.core_axes
        }

        # Select the relevant subset from X using dynamic vectorized selection
        seed_pixels = X.sel(**seed_das)

        ks = xr.apply_ufunc(
            self.ks_kernel,
            seed_pixels,
            input_core_dims=[self.iter_axis],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        ks_computed = ks < self.significance_threshold
        seeds["mask_ks"] = ks_computed.compute().values

        return seeds

    def transform(self, X: xr.DataArray, y: pd.DataFrame):

        return self.transform_kernel(X, y)

    @staticmethod
    def ks_kernel(arr: np.ndarray) -> float:
        """
        Computes the p-value of the Kolmogorov-Smirnov test against the normal distribution
        after z-score normalization. Returns 0.0 if the array is constant.
        """
        if np.all(arr == arr[0]):
            return 0.0  # Reject null hypothesis if data is constant
        standardized = zscore(arr)
        return kstest(standardized, "norm").pvalue

    def fit_transform_shared_preprocessing(self, X, y):
        pass
