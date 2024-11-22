from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture
from .base import BaseFilter


def gmm_refine(
    self,
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    quantiles: Tuple[float, float] = (0.1, 99.9),
    n_components: int = 2,
    valid_components: int = 1,
    mean_mask: bool = True,
) -> Tuple[pd.DataFrame, xr.DataArray, GaussianMixture]:
    """
    Filter seeds by fitting a GMM to peak-to-peak values.

    This function assumes that the distribution of peak-to-peak values of
    fluorescence across all seeds can be modeled by a Gaussian Mixture Model (GMM)
    with different means. It computes peak-to-peak values for all the seeds, then
    fits a GMM with `n_components` to the distribution, and filters out the seeds
    belonging to the `n_components - valid_components` number of gaussians with
    lower means.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data. Should have dimensions "spatial" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    quantiles : tuple, optional
        Percentiles to use to compute the peak-to-peak values. For a given seed
        with corresponding fluorescent fluctuation `f`, the peak-to-peak value
        for that seed is computed as `np.percentile(f, q[1]) - np.percentile(f, q[0])`.
        By default `(0.1, 99.9)`.
    n_components : int, optional
        Number of components (Gaussians) in the GMM model. By default `2`.
    valid_components : int, optional
        Number of components (Gaussians) to be considered as modeling the
        distribution of peak-to-peak values of valid seeds. Should be smaller
        than `n_components`. By default `1`.
    mean_mask : bool, optional
        Whether to apply an additional criterion where a seed is valid only if its
        peak-to-peak value exceeds the mean of the lowest Gaussian distribution.
        Only useful in corner cases where the distribution of the Gaussian
        heavily overlaps. By default `True`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_gmm",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds`, it will be overwritten.
    varr_pv : xr.DataArray
        The computed peak-to-peak values for each seed.
    gmm : GaussianMixture
        The fitted GMM model object.

    See Also
    -------
    sklearn.mixture.GaussianMixture
    """
    # Select the spatial points corresponding to the seeds
    spatial_coords = seeds[self.core_axes].apply(tuple, axis=1).tolist()
    varr_sub = varr.sel(spatial=spatial_coords)

    # Compute both percentiles in a single quantile call
    quantiles = varr_sub.quantile(
        q=quantiles, dim=self.iter_axis, interpolation="linear"
    )
    varr_valley = quantiles.sel(q=quantiles[0])
    varr_peak = quantiles.sel(q=quantiles[1])
    varr_pv = varr_peak - varr_valley
    varr_pv = varr_pv.compute()

    # Fit the Gaussian Mixture Model
    data = varr_pv.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    # Identify valid components based on sorted means
    sorted_indices = np.argsort(gmm.means_.flatten())
    valid_component_indices = sorted_indices[-valid_components:]

    # Predict cluster assignments and determine validity
    cluster_labels = gmm.predict(data)
    is_valid = np.isin(cluster_labels, valid_component_indices)

    # Apply mean mask if required
    if mean_mask:
        lowest_mean = np.min(gmm.means_)
        mean_mask_condition = data.flatten() > lowest_mean
        is_valid &= mean_mask_condition

    # Update the seeds DataFrame with the mask
    seeds = seeds.copy()
    seeds["mask_gmm"] = is_valid

    return seeds, varr_pv, gmm
