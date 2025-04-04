from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.mixture import GaussianMixture

from ..signal_processing import median_clipper
from .base import BaseFilter


@dataclass
class PNRFilter(BaseFilter):
    """Filter seeds based on peak-to-noise ratio (PNR) analysis.

    This filter analyzes the peak-to-noise ratio of fluorescence signals to identify valid seeds.
    Seeds are considered valid if their PNR exceeds a threshold or if they belong to high-PNR
    clusters determined by Gaussian Mixture Model analysis.

    The filter assumes that valid seeds exhibit higher peak-to-noise ratios compared to noise.

    Notes
    -----
    The filtering process follows these steps:
    1. Applies optional median filtering to the input signals
    2. Computes peak-to-noise ratio for each seed location
    3. Either:
        a) Marks seeds as valid if PNR exceeds pnr_threshold, or
        b) Uses GMM clustering to identify high-PNR seeds if pnr_threshold is None
    """

    cutoff_frequency_ratio: float = 0.06
    """Cutoff frequency ratio for high-pass filtering, must be between 0 and 0.5 (Nyquist frequency)."""
    pnr_threshold: float | None = 1.0
    """Threshold for deciding valid seeds based on PNR."""
    auto_pnr_threshold: bool = False
    """Whether to automatically determine the PNR threshold using GMM clustering. If True, pnr_threshold is ignored."""
    quantile_floor: float = 5.0
    """Lower quantile threshold for PNR calculation."""
    quantile_ceil: float = 95.0
    """Upper quantile threshold for PNR calculation."""
    filter_window_size: int | None = None
    """Window size for median filtering. If None, no filtering is applied."""
    pnr_: xr.DataArray = field(init=False)
    """Computed peak-to-noise ratios for the input data."""
    valid_pnr_: np.ndarray = field(init=False)
    """Boolean mask indicating valid seeds based on PNR analysis."""
    gmm_: GaussianMixture = field(init=False)
    """Fitted Gaussian Mixture Model for PNR clustering."""
    _stateless: ClassVar[bool] = True
    """Whether the filter is stateless. Always True for this filter."""

    def __post_init__(self) -> None:
        if not 0 < self.cutoff_frequency_ratio <= 0.5:
            raise ValueError("cutoff_frequency must be between 0 and 0.5 (Nyquist frequency).")

        if self.quantile_floor >= self.quantile_ceil:
            raise ValueError("quantile_floor must be smaller than quantile_ceil")

    @property
    def quantiles(self) -> tuple[float, float]:
        return self.quantile_floor, self.quantile_ceil

    def fit_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
        pass

    def fit_transform_shared_preprocessing(self, X: xr.DataArray, seeds: pd.DataFrame) -> None:
        pass

    def transform_kernel(self, X: xr.DataArray, seeds: pd.DataFrame) -> pd.DataFrame:
        """
        Transform seeds by filtering based on peak-to-noise ratio analysis.

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
            Seeds DataFrame with an additional boolean column 'mask_pnr' indicating valid seeds.
            Seeds are marked as valid if their peak-to-noise ratio exceeds the threshold
            (if pnr_threshold is set) or if they belong to the highest-PNR cluster
            (if pnr_threshold is None).

        Notes
        -----
        This method:
        1. Optionally applies median filtering to the seed signals
        2. Computes peak-to-noise ratio for each seed
        3. Determines valid seeds either by:
           - Thresholding against pnr_threshold if set
           - Using GMM clustering to find highest-PNR cluster if no pnr_threshold
        4. Returns seeds DataFrame with additional mask column

        The peak-to-noise ratio helps identify signal-like behavior by comparing
        signal peaks to noise levels, with higher ratios indicating stronger signals.
        """
        # Dynamically create a dictionary of DataArrays for each core axis
        seed_das: dict[str, xr.DataArray] = {
            axis: xr.DataArray(seeds[axis].values, dims="seeds") for axis in self.core_axes
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
                "cutoff_frequency": self.cutoff_frequency_ratio,
                "quantiles": self.quantiles,
            },
            output_dtypes=[float],
        ).compute()

        if self.auto_pnr_threshold:
            valid_pnr_ = np.nan_to_num(pnr.values.reshape(-1, 1))
            mask = self._find_highest_pnr_cluster_gmm(valid_pnr_)
        else:
            mask = pnr > self.pnr_threshold

        seeds["mask_pnr"] = mask.values

        return seeds

    def _find_highest_pnr_cluster_gmm(self, pnr: np.ndarray) -> np.ndarray:
        """Find seeds with high peak-to-noise ratio using Gaussian Mixture Model clustering.

        This method fits a 2-component Gaussian Mixture Model to the distribution of peak-to-noise
        ratios and identifies seeds belonging to the component with the higher mean value.

        Parameters
        ----------
        pnr : numpy.ndarray
            Array of peak-to-noise ratio values, shape (n_seeds, 1)

        Returns
        -------
        numpy.ndarray
            Boolean mask indicating seeds belonging to the high PNR component.
            True values correspond to seeds with high peak-to-noise ratios.

        Notes
        -----
        The method assumes that peak-to-noise ratios follow a bimodal distribution, with:
        - One component representing low PNR values (noise-like seeds)
        - One component representing high PNR values (signal-like seeds)

        Seeds are marked as valid if they belong to the component with the higher mean value.
        """
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
        quantiles: tuple[float, float],
        filter_pass: Literal["high", "low"] = "high",
    ) -> float:
        """Calculate the peak-to-noise ratio of a signal using FFT filtering.

        This method computes the ratio between the peak-to-peak amplitude of the original signal
        and the peak-to-peak amplitude of the filtered signal (noise). The filtering is performed
        using FFT to isolate specific frequency components.

        Parameters
        ----------
        arr : numpy.ndarray
            1D array containing the signal to analyze
        cutoff_frequency : float
            Normalized frequency (0 to 1) used as cutoff for filtering
        quantiles : tuple of float
            (lower, upper) quantiles used for computing peak-to-peak amplitudes
        filter_pass : {'high', 'low'}, default='high'
            Type of frequency filter to apply:
            - 'high': keeps frequencies above cutoff (isolates high-freq noise)
            - 'low': keeps frequencies below cutoff (isolates low-freq noise)

        Returns
        -------
        float
            Peak-to-noise ratio. Higher values indicate stronger signal relative to noise.
            Returns infinity if noise amplitude is zero.

        Notes
        -----
        The method:
        1. Computes peak-to-peak amplitude of original signal using quantiles
        2. Applies FFT-based filtering to isolate noise components
        3. Computes peak-to-peak amplitude of filtered signal (noise)
        4. Returns ratio of original amplitude to noise amplitude
        """

        # Compute peak-to-peak (ptp) before filtering
        peak_to_peak = np.percentile(arr, quantiles[1]) - np.percentile(arr, quantiles[0])

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
        return float(peak_to_peak / peak_to_peak_noise if peak_to_peak_noise != 0 else np.inf)
