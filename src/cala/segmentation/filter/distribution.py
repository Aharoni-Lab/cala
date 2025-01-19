from dataclasses import dataclass
from typing import Dict, ClassVar, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


from .base import BaseFilter


@dataclass
class DistributionFilter(BaseFilter):
    """
    Filter the seeds using a distribution test.

    This function assume that the valid seeds’ fluorescence across frames
    notionally follows a bimodal distribution: with a large normal distribution
    representing baseline activity, and a second peak representing when the
    seed/cell is active.

    """

    num_peaks: Optional[int] = None
    _stateless: ClassVar[bool] = True

    def fit_kernel(self, X, seeds):
        pass

    def fit_transform_shared_preprocessing(self, X, seeds):
        pass

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame):
        """
        Returns : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_ks",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
        """
        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: Dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds")
            for axis in self.core_axes
        }

        # Select the relevant subset from X using dynamic vectorized selection
        seed_pixels = X.sel(**seed_das)

        n_components = xr.apply_ufunc(
            self.min_ic_components,
            seed_pixels,
            input_core_dims=[[self.iter_axis]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        if self.num_peaks:
            n_components_computed = n_components >= self.num_peaks
        else:
            n_components_computed = n_components > 1
        seeds["mask_dist"] = n_components_computed.compute().values

        return seeds

    @staticmethod
    def min_ic_components(arr: np.ndarray, max_components=5):
        """
        Fit GMMs with components = 1..max_components to 'data',
        return the model with the best (lowest) BIC.
        """
        arr = arr.reshape(-1, 1)
        best_model = None
        lowest_aic = np.inf

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k)
            gmm.fit(arr)
            aic = gmm.aic(arr)
            if aic < lowest_aic:
                lowest_aic = aic
                best_model = gmm

        return best_model.n_components
