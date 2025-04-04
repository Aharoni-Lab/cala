from dataclasses import dataclass

import pandas as pd
import xarray as xr
from skimage.feature import peak_local_max
from skimage.morphology import disk

from .base import BaseDetector


@dataclass
class MaxProjection(BaseDetector):
    """
    Detect cells using maximum intensity projection and local maxima detection.

    This detector first computes the maximum intensity projection of the input data
    along the iteration axis, then finds local maxima in the resulting 2D image
    using a disk-shaped structuring element.

    Parameters
    ----------
    core_axes : List[str]
        Names of the spatial dimensions in the data (e.g., ["height", "width"])
    iter_axis : str
        Name of the iteration dimension (e.g., "frames" for time series)
    """

    # Radius for local maxima detection.
    local_max_radius: int = 10
    # Intensity threshold for peak detection.
    intensity_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        super().__post_init__()
        if self.local_max_radius <= 0:
            raise ValueError("local_max_radius must be positive")
        if self.intensity_threshold <= 0:
            raise ValueError("intensity_threshold must be positive")

    def fit_kernel(self, X: xr.DataArray) -> list[xr.DataArray]:
        """
        Compute max projections for each window.
        """
        return [
            X.isel({self.iter_axis: indices}).max(dim=self.iter_axis)
            for indices in self.window_indices_
        ]

    def transform_kernel(self, max_projection: xr.DataArray) -> pd.DataFrame:
        """
        Detect cells using local maxima in the max projection.

        Parameters
        ----------
        X : xr.DataArray
            Input data array.

        Returns
        -------
        pd.DataFrame
            DataFrame containing detected cell positions.
        """

        # Create structuring element for local maxima detection
        selem = disk(self.local_max_radius)

        # Find local maxima
        peaks = peak_local_max(
            max_projection.values,
            footprint=selem,
            threshold_abs=self.intensity_threshold,
            min_distance=1,
        )

        # Convert peak positions to DataFrame
        seeds = pd.DataFrame(peaks, columns=self.core_axes)
        return seeds
