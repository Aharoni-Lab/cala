from typing import Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import xarray as xr
from ..signal_processing import med_baseline
from .base import BaseFilter


def pnr_refine(
    self,
    varr: xr.DataArray,
    seeds: pd.DataFrame,
    noise_freq: float = 0.25,
    thres: float | str = 1.5,
    q: Tuple[float, float] = (0.1, 99.9),
    med_wnd: Optional[int] = None,
) -> Tuple[pd.DataFrame, xr.DataArray, Optional[GaussianMixture]]:
    """
    Filter seeds by thresholding peak-to-noise ratio.

    For each input seed, the noise is defined as a high-pass filtered fluorescence
    trace of the seed. The peak-to-noise ratio (pnr) of that seed is then
    defined as the ratio between the peak-to-peak value of the original
    fluorescence trace and that of the noise trace. Optionally, if abrupt
    changes in baseline fluorescence are expected, then the baseline can be
    estimated by median-filtering the fluorescence trace and subtracted from the
    original trace before computing the peak-to-noise ratio. In addition, if a
    hard threshold of pnr is not desired, then a Gaussian Mixture Model with 2
    components can be fitted to the distribution of pnr across all seeds, and
    only seeds with pnr belonging to the higher-mean Gaussian will be considered
    valid.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data, should have dimensions "height", "width", and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    noise_freq : float, optional
        Cut-off frequency for the high-pass filter used to define noise,
        specified as a fraction of the sampling frequency. By default `0.25`.
    thres : Union[float, str], optional
        Threshold of the peak-to-noise ratio. If `"auto"`, then a Gaussian Mixture
        Model will be fit to the distribution of pnr. By default `1.5`.
    q : tuple, optional
        Percentiles to use to compute the peak-to-peak values. For a given
        fluorescence fluctuation `f`, the peak-to-peak value for that seed is
        computed as `np.percentile(f, q[1]) - np.percentile(f, q[0])`. By
        default `(0.1, 99.9)`.
    med_wnd : int, optional
        Size of the median filter window to remove baseline. If `None`, then no
        filtering will be done. By default `None`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_pnr",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds`, it will be overwritten.
    pnr : xr.DataArray
        The computed peak-to-noise ratio for each seed.
    gmm : GaussianMixture, optional
        The GMM model object fitted to the distribution of pnr. Will be `None`
        unless `thres` is `"auto"`.
    """
    # Determine optimal chunk size to prevent memory issues
    total_seeds = len(seeds)
    chk_size = min(max(int(total_seeds / 128), 1), 100)

    # Split seeds into chunks and select corresponding data
    seed_indices = np.arange(total_seeds)
    seed_groups = np.array_split(seed_indices, np.ceil(total_seeds / chk_size))

    varr_sub_chunks = [
        varr.sel(
            height=seeds.iloc[group]["height"].values,
            width=seeds.iloc[group]["width"].values,
        )
        for group in seed_groups
    ]

    varr_sub = xr.concat(varr_sub_chunks, dim="seed_index")

    # Apply median filtering to remove baseline if specified
    if med_wnd is not None:
        varr_filtered = xr.apply_ufunc(
            med_baseline,
            varr_sub,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            kwargs={"wnd": med_wnd},
            output_dtypes=[varr.dtype],
        )
    else:
        varr_filtered = varr_sub

    # Compute peak-to-noise ratio using the provided pnr_kernel function
    pnr = xr.apply_ufunc(
        pnr_kernel,
        varr_filtered,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        kwargs={"freq": noise_freq, "q": q},
        output_dtypes=[float],
    ).compute()

    # Initialize GMM as None
    gmm = None

    if thres == "auto":
        # Fit Gaussian Mixture Model to pnr distribution
        valid_pnr = np.nan_to_num(pnr.values.reshape(-1, 1))
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(valid_pnr)

        # Identify the component with the higher mean
        component_means = gmm.means_.flatten()
        high_mean_component = np.argmax(component_means)

        # Predict cluster labels and determine valid seeds
        cluster_labels = gmm.predict(valid_pnr)
        is_valid = cluster_labels == high_mean_component
        mask = is_valid
    else:
        # Apply a hard threshold to determine valid seeds
        mask = pnr > thres

    # Assign the mask to the seeds DataFrame
    seeds = seeds.copy()
    seeds["mask_pnr"] = mask.values

    return seeds, pnr, gmm


def pnr_kernel(a: np.ndarray, freq: float, q: tuple) -> float:
    """
    Compute peak-to-noise ratio of a given timeseries.

    Parameters
    ----------
    a : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency of the high-pass filtering used to define noise.
    q : tuple
        Percentile used to compute peak-to-peak values.

    Returns
    -------
    pnr : float
        Peak-to-noise ratio.

    See Also
    -------
    pnr_refine : for definition of peak-to-noise ratio
    """
    ptp = np.percentile(a, q[1]) - np.percentile(a, q[0])
    a = filt_fft(a, freq, btype="high")
    ptp_noise = np.percentile(a, q[1]) - np.percentile(a, q[0])
    return ptp / ptp_noise


def filt_fft(x: np.ndarray, freq: float, btype: str) -> np.ndarray:
    """
    Filter 1d timeseries by zero-ing bands in the fft signal.

    Parameters
    ----------
    x : np.ndarray
        Input timeseries.
    freq : float
        Cut-off frequency.
    btype : str
        Either `"low"` or `"high"` specify low or high pass filtering.

    Returns
    -------
    x_filt : np.ndarray
        Filtered timeseries.
    """
    _T = len(x)
    if btype == "low":
        zero_range = slice(int(freq * _T), None)
    elif btype == "high":
        zero_range = slice(None, int(freq * _T))
    xfft = numpy_fft.rfft(x)
    xfft[zero_range] = 0
    return numpy_fft.irfft(xfft, len(x))
