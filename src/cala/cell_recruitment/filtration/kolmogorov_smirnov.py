# gmm_filter pnr_filter intensity_filter ks_filter

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pyfftw.interfaces.numpy_fft as numpy_fft
import xarray as xr
from scipy.stats import kstest, zscore
from .base import BaseFilter


def ks_refine(self, varr: xr.DataArray, seeds: pd.DataFrame, sig=0.01) -> pd.DataFrame:
    """
    Filter the seeds using Kolmogorov-Smirnov (KS) test.

    This function assume that the valid seeds’ fluorescence across frames
    notionally follows a bimodal distribution: with a large normal distribution
    representing baseline activity, and a second peak representing when the
    seed/cell is active. KS allows to discard the seeds where the
    null-hypothesis (i.e. the fluorescence intensity is simply a normal
    distribution) is rejected at `sig`.

    Parameters
    ----------
    varr : xr.DataArray
        Input movie data. Should have dimensions "height", "width" and "frame".
    seeds : pd.DataFrame
        The input over-complete set of seeds to be filtered.
    sig : float, optional
        The significance threshold to reject null-hypothesis. By default `0.01`.

    Returns
    -------
    seeds : pd.DataFrame
        The resulting seeds dataframe with an additional column "mask_ks",
        indicating whether the seed is considered valid by this function. If the
        column already exists in input `seeds` it will be overwritten.
    """
    print("selecting seeds")
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(
            height=seed_sub["height"].to_xarray(),
            width=seed_sub["width"].to_xarray(),
        )
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
    print("performing KS test")
    ks = xr.apply_ufunc(
        ks_kernel,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    ks = (ks < sig).to_pandas().rename("mask_ks")
    seeds["mask_ks"] = ks
    return seeds


def ks_kernel(a: np.ndarray) -> float:
    """
    Perform KS test on input and return the p-value.

    Parameters
    ----------
    a : np.ndarray
        Input data.

    Returns
    -------
    p : float
        The p-value of the KS test.

    See Also
    -------
    scipy.stats.kstest
    """
    a = zscore(a)
    return kstest(a, "norm")[1]
