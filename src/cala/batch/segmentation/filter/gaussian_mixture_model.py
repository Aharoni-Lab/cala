# from dataclasses import dataclass, field
#
# import numpy as np
# import pandas as pd
# import xarray as xr
# from sklearn.mixture import GaussianMixture
#
# from .base import BaseFilter
#
#
# @dataclass
# class GMMFilter(BaseFilter):
#     """Filter seeds based on their fluorescence amplitude using Gaussian Mixture Models.
#
#     This filter analyzes the amplitude of fluorescence changes for each seed using
#     a Gaussian Mixture Model (GMM). Valid seeds are expected to show significant
#     fluorescence changes between their peak and valley values, characterized by:
#     - Higher amplitude components representing active cells
#     - Lower amplitude components representing noise or background
#
#     Notes
#     -----
#     The filter works by:
#     1. Computing peak-to-valley amplitude for each seed
#     2. Fitting a GMM to these amplitudes
#     3. Identifying components with highest means as valid cells
#     4. Optionally applying additional mean-based masking
#     """
#
#     quantile_floor: float = 0.1
#     """Lower quantile for computing fluorescence amplitude."""
#     quantile_ceil: float = 0.99
#     """Upper quantile for computing fluorescence amplitude."""
#     num_components: int = 2
#     """Number of Gaussian components to fit."""
#     num_valid_components: int = 1
#     """Number of highest-mean components to consider as valid."""
#     mean_mask: bool = True
#     """Whether to apply additional masking based on component means."""
#     seed_amplitude_: np.ndarray = field(init=False)
#     """Amplitude of fluorescence changes for each seed."""
#     gmm_: GaussianMixture = field(init=False)
#     """Gaussian Mixture Model fitted to seed amplitudes."""
#     valid_component_indices_: np.ndarray = field(init=False)
#     """Indices of valid components based on highest means."""
#
#     def __post_init__(self):
#         """Validate parameters and initialize GMM model."""
#         if self.quantile_floor >= self.quantile_ceil:
#             raise ValueError("quantile_floor must be smaller than quantile_ceil")
#         if self.quantile_floor < 0 or self.quantile_ceil > 1:
#             raise ValueError("quantiles must be between 0 and 1")
#         self.gmm_ = GaussianMixture(random_state=42)
#
#     @property
#     def quantiles(self):
#         """Return quantile values used for amplitude calculation.
#
#         Returns
#         -------
#         tuple
#             (quantile_floor, quantile_ceil) values
#         """
#         return self.quantile_floor, self.quantile_ceil
#
#     def fit_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
#         """Fit a Gaussian Mixture Model to seed amplitudes and identify valid components.
#
#         Parameters
#         ----------
#         X : xarray.DataArray
#             Input fluorescence data with dimensions matching core_axes and iter_axis.
#             Contains the time series data for each spatial point.
#         seeds : pandas.DataFrame
#             Seed coordinates in columns matching core_axes.
#             Each row represents a seed location to analyze.
#
#         Returns
#         -------
#         None
#             Updates internal state by fitting GMM and identifying valid components.
#
#         Notes
#         -----
#         This method performs the following steps:
#         1. Configures the GMM with the specified number of components
#         2. Fits the GMM to the pre-computed seed amplitudes stored in self.seed_amplitude_
#         3. Identifies the components with the highest means as valid components
#         4. Stores the indices of valid components in self.valid_component_indices_
#
#         See Also
#         --------
#         transform_kernel : Uses the fitted GMM to filter seeds
#         fit_transform_shared_preprocessing : Computes seed amplitudes used by this method
#         """
#         self.gmm_.set_params(n_components=self.num_components)
#         self.gmm_.fit(self.seed_amplitude_)
#
#         self.valid_component_indices_ = np.argsort(self.gmm_.means_.reshape(-1))[
#             -self.num_valid_components :
#         ]  # get the indices of the components with the highest means
#
#     def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
#         """Filter seeds based on their GMM component assignments.
#
#         Parameters
#         ----------
#         X : xarray.DataArray
#             Input fluorescence data with dimensions matching core_axes and iter_axis.
#             Contains the time series data for each spatial point.
#         seeds : pandas.DataFrame
#             Seed coordinates in columns matching core_axes.
#             Each row represents a seed location to analyze.
#
#         Returns
#         -------
#         pandas.DataFrame
#             Seeds DataFrame with an additional boolean column 'mask_gmm' indicating valid seeds.
#             Seeds are marked as valid if they belong to high-mean GMM components.
#
#         Notes
#         -----
#         Seeds are marked as valid if:
#         1. They belong to one of the highest-mean GMM components
#         2. (Optional) Their amplitude exceeds the lowest component mean
#
#         See Also
#         --------
#         fit_kernel : Fits the GMM model used by this method
#         fit_transform_shared_preprocessing : Computes seed amplitudes used by this method
#
#         Examples
#         --------
#         >>> filter = GMMFilter()
#         >>> filter.fit(X, seeds)
#         >>> filtered_seeds = filter.transform(X, seeds)
#         >>> valid_seeds = filtered_seeds[filtered_seeds['mask_gmm']]
#         """
#         # Predict cluster assignments and determine validity
#         cluster_labels = self.gmm_.predict(self.seed_amplitude_)
#         is_valid = np.isin(cluster_labels, self.valid_component_indices_)
#
#         # Apply mean mask if required
#         if self.mean_mask:
#             lowest_mean = np.min(self.gmm_.means_)
#             mean_mask_condition = self.seed_amplitude_.flatten() > lowest_mean
#             is_valid &= mean_mask_condition
#
#         # Update the seeds DataFrame with the mask
#         seeds["mask_gmm"] = is_valid
#
#         return seeds
#
#     def fit_transform_shared_preprocessing(self, X: xr.DataArray, seeds):
#         """Compute peak-to-valley amplitude for each seed.
#
#         Parameters
#         ----------
#         X : xarray.DataArray
#             Input fluorescence data with dimensions matching core_axes and iter_axis.
#             Contains the time series data for each spatial point.
#         seeds : pandas.DataFrame
#             Seed coordinates in columns matching core_axes.
#             Each row represents a seed location to analyze.
#
#         Returns
#         -------
#         None
#             Updates self.seed_amplitude_ with computed amplitudes.
#
#         Attributes
#         ----------
#         seed_amplitude_ : numpy.ndarray
#             Array of shape (n_seeds, 1) containing peak-to-valley amplitudes
#             for each seed.
#
#         Notes
#         -----
#         This method:
#         1. Extracts fluorescence traces for each seed
#         2. Computes quantile-based peak and valley values
#         3. Calculates peak-to-valley amplitude
#
#         The amplitude is computed as the difference between upper and lower
#         quantiles of the fluorescence trace, where the quantiles are specified
#         by self.quantiles.
#
#         See Also
#         --------
#         fit_kernel : Fits GMM using the computed amplitudes
#         transform_kernel : Uses the computed amplitudes for filtering
#         """
#         # Select the spatial points corresponding to the seeds
#         spatial_coords = seeds[self.core_axes].apply(tuple, axis=1).tolist()
#         X = X.stack({self.spatial_axis: self.core_axes})
#         seed_pixels = X.sel({self.spatial_axis: spatial_coords})
#
#         # Compute both percentiles in a single quantile call
#         quantiles = seed_pixels.quantile(
#             q=self.quantiles,
#             dim=self.iter_axis,
#             method="linear",
#         )
#         seed_valley = quantiles.sel(quantile=self.quantiles[0])
#         seed_peak = quantiles.sel(quantile=self.quantiles[1])
#         self.seed_amplitude_ = (seed_peak - seed_valley).compute().values.reshape(-1, 1)
