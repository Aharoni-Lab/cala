from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from .base import BaseFilter


@dataclass
class DistributionFilter(BaseFilter):
    """
    Filter seeds based on their fluorescence distribution characteristics.

    This filter analyzes the distribution of fluorescence values across frames for each seed.
    Valid seeds are expected to show a non-normal distribution pattern, characterized by:
    - A large normal component representing baseline activity
    - A long tail component representing periods of cellular activity

    Notes
    -----
    The filter uses Gaussian Mixture Models to detect the number of components
    in each seed's fluorescence distribution. Seeds are considered valid if they
    exhibit at least the specified number of components, indicating the presence
    of both baseline and active states.
    """

    num_peaks: int = 2
    """Number of peaks to detect in the fluorescence distribution."""
    _stateless: ClassVar[bool] = True
    """Indicates the filter is stateless and does not require fitting. True for this filter."""

    def fit_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
        pass

    def fit_transform_shared_preprocessing(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
        pass

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
        """Transform the seeds by filtering based on distribution analysis.

        Parameters
        ----------
        X : xarray.DataArray
            Fluorescence data with dimensions matching core_axes and iter_axis.
            Contains the time series data for each spatial point.
        seeds : pandas.DataFrame
            Seed coordinates in columns matching core_axes.
            Each row represents a seed location to analyze.

        Returns
        -------
        pandas.DataFrame
            Seeds DataFrame with an additional boolean column 'mask_dist' indicating valid seeds.
            Seeds are marked as valid if their fluorescence distribution has at least the
            specified number of components.

        Notes
        -----
        This method analyzes each seed's fluorescence distribution across frames using
        Gaussian Mixture Models (GMM). It determines if the distribution has the expected
        number of components (peaks) by finding the optimal number of GMM components using
        AIC criterion. Seeds are marked as valid if their fluorescence distribution has at
        least the specified number of components. The method assumes that noise follows a
        normal distribution.
        """
        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds") for axis in self.core_axes
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
    def min_ic_components(arr: np.ndarray, max_components: int = 5) -> int:
        """
        Find the optimal number of Gaussian components using AIC criterion.

        Parameters
        ----------
        arr : numpy.ndarray
            1D array of values to fit GMM components to
        max_components : int, optional
            Maximum number of components to try fitting, by default 5

        Returns
        -------
        int
            Optimal number of components according to AIC criterion

        Notes
        -----
        This method fits multiple Gaussian Mixture Models with increasing numbers
        of components (from 1 to max_components) and selects the model with the
        lowest Akaike Information Criterion (AIC) score. The number of components
        in the best-fitting model is returned.
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

        if best_model is None:
            return 1  # Default to 1 component if no model was fit
        return int(best_model.n_components)
