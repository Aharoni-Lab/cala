from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from cala.segmentation.signal_processing import median_clipper
from .base import BaseFilter


@dataclass
class PNRFilter(BaseFilter):
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
    """

    cutoff_frequency: float = 0.06
    pnr_threshold: Optional[float] = 1.0
    quantile_floor: float = 5.0
    quantile_ceil: float = 95.0
    filter_window_size: Optional[int] = None
    pnr_: xr.DataArray = field(init=False)
    valid_pnr_: np.ndarray = field(init=False)
    gmm_: GaussianMixture = field(init=False)
    _stateless: ClassVar[bool] = True
    """
    pnr_threshold: if None, finds it automatically.
    """

    def __post_init__(self):
        if not 0 < self.cutoff_frequency <= 0.5:
            raise ValueError(
                "cutoff_frequency must be between 0 and 0.5 (Nyquist frequency)."
            )

        if self.quantile_floor >= self.quantile_ceil:
            raise ValueError("quantile_floor must be smaller than quantile_ceil")

    @property
    def quantiles(self):
        return self.quantile_floor, self.quantile_ceil

    def fit_kernel(self, X, seeds) -> None:
        pass

    def fit_transform_shared_preprocessing(self, X, seeds):
        pass

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame):
        if hasattr(X, "air") and X.air.chunks is None:
            X = X.chunk(auto=True)

        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: Dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds")
            for axis in self.core_axes
        }

        # Select the relevant subset from X using vectorized selection
        seed_pixels = X.sel(**seed_das)

        if self.filter_window_size is not None:
            seeds_filtered = xr.apply_ufunc(
                median_clipper,
                seed_pixels,
                input_core_dims=[self.iter_axis],
                output_core_dims=[self.iter_axis],
                vectorize=True,
                dask="parallelized",
                kwargs={"window_size": self.filter_window_size},
                output_dtypes=[seed_pixels.dtype],
            )
        else:
            seeds_filtered = seed_pixels

        # Compute peak-to-noise ratio
        pnr = xr.apply_ufunc(
            self.pnr_kernel,
            seeds_filtered,
            input_core_dims=[[self.iter_axis]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            kwargs={
                "cutoff_frequency": self.cutoff_frequency,
                "quantiles": self.quantiles,
            },
            output_dtypes=[float],
        ).compute()

        if self.pnr_threshold is None:
            valid_pnr_ = np.nan_to_num(pnr.values.reshape(-1, 1))
            mask = self._find_highest_pnr_cluster_gmm(valid_pnr_)
        else:
            mask = pnr > self.pnr_threshold

        seeds["mask_pnr"] = mask.values

        return seeds

    def _find_highest_pnr_cluster_gmm(self, pnr):
        # Fit Gaussian Mixture Model to pnr distribution
        self.gmm_ = GaussianMixture(n_components=2, random_state=42)
        self.gmm_.fit(pnr)

        # Identify the component with the higher mean
        component_means = self.gmm_.means_.flatten()
        high_mean_components = np.argmax(component_means)

        # Predict cluster labels and determine valid seeds
        cluster_labels = self.gmm_.predict(pnr)
        return cluster_labels == high_mean_components

    @staticmethod
    def pnr_kernel(
        arr: np.ndarray,
        cutoff_frequency: float,
        quantiles: tuple,
        filter_pass: Literal["high", "low"] = "high",
    ) -> float:
        """
        Compute the Peak-to-Noise Ratio (PNR) of a given timeseries after applying a high-pass or low-pass filter.

        Parameters
        ----------
        arr : np.ndarray
            Input timeseries.
        cutoff_frequency : float
            Cut-off frequency as a fraction of the sampling rate (0 < freq < 0.5).
        quantiles : tuple of float
            Percentiles used to compute peak-to-peak values (e.g., (5, 95)).
        filter_pass : str, optional
            Type of filter to apply: "high" for high-pass or "low" for low-pass filtering. Default is "high".

        Returns
        -------
        float
            Peak-to-noise ratio.
        """

        # Compute peak-to-peak (ptp) before filtering
        peak_to_peak = np.percentile(arr, quantiles[1]) - np.percentile(
            arr, quantiles[0]
        )

        # Apply FFT-based filter
        _T = len(arr)
        cutoff_bin = int(cutoff_frequency * _T)

        # Perform real FFT
        frequency_composition = np.fft.rfft(arr)

        # Zero out the specified frequency bands
        if filter_pass == "low":
            frequency_composition[cutoff_bin:] = 0
        elif filter_pass == "high":
            frequency_composition[:cutoff_bin] = 0

        # Perform inverse real FFT to obtain the filtered signal
        filtered_arr = np.fft.irfft(frequency_composition, n=_T)

        # Compute peak-to-peak (ptp_noise) after filtering
        peak_to_peak_noise = np.percentile(filtered_arr, quantiles[1]) - np.percentile(
            filtered_arr, quantiles[0]
        )

        # Calculate and return the Peak-to-Noise Ratio
        return peak_to_peak / peak_to_peak_noise if peak_to_peak_noise != 0 else np.inf
