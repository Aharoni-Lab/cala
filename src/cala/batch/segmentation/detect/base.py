from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BaseDetector(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for cell detection algorithms.

    This class provides a common interface for detecting cells in fluorescence microscopy data.
    It follows the sklearn transformer pattern and supports both fitting (optional) and
    transformation operations.
    """

    # Names of the spatial dimensions in the data (e.g., ["height", "width"])
    core_axes: list[str]
    # Name of the iteration dimension (e.g., "frames" for time series)
    iter_axis: str
    # Either "rolling" or "random". Controls whether to use rolling window
    # or random sampling of frames to construct chunks.
    method: Literal["rolling", "random"] = "rolling"
    # Number of frames in each chunk/window.
    chunk_size: int = 500
    # Number of frames between the center of each chunk when stepping through
    # the data with rolling windows. Only used if method is "rolling".
    step_size: int = 200
    # Number of chunks to sample randomly. Only used if method is "random".
    num_chunks: int = 100
    # Indices for each window
    window_indices_: list[slice] = field(default_factory=list, init=False)
    # Projections for each window
    window_projections_: list[xr.DataArray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.method not in ["rolling", "random"]:
            raise ValueError("method must be either 'rolling' or 'random'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")

        if not isinstance(self.core_axes, list) or not all(
            isinstance(x, str) for x in self.core_axes
        ):
            raise ValueError("core_axes must be a list of strings")
        if not isinstance(self.iter_axis, str):
            raise ValueError("iter_axis must be a string")

    def _get_window_indices(self, num_frames: int) -> list:
        """Get indices for each window based on the method."""
        if self.method == "rolling":
            num_steps = max(int(np.ceil((num_frames - self.chunk_size) / self.step_size)) + 1, 1)
            start_indices = (np.arange(num_steps) * self.step_size).astype(int)
            return [
                slice(start, min(start + self.chunk_size, num_frames)) for start in start_indices
            ]
        else:  # random
            return [
                np.random.choice(num_frames, size=min(self.chunk_size, num_frames), replace=False)
                for _ in range(self.num_chunks)
            ]

    def fit(self, X: xr.DataArray, y: Any = None) -> "BaseDetector":
        """
        Compute max projections for each window.

        Parameters
        ----------
        X : xr.DataArray
            Input data array.
        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self : WindowDetector
            Returns self for method chaining.
        """
        num_frames = X.sizes[self.iter_axis]
        self.window_indices_ = self._get_window_indices(num_frames)

        self.window_projections_ = self.fit_kernel(X)

        return self

    @abstractmethod
    def fit_kernel(self, X: xr.DataArray) -> list[xr.DataArray]:
        pass

    def transform(self, X: xr.DataArray) -> pd.DataFrame:
        """
        Detect cells by combining detections from all windows.

        Parameters
        ----------
        X : xr.DataArray
            Input data array.

        Returns
        -------
        pd.DataFrame
            DataFrame containing detected cell positions.
        """
        if not self.window_projections_:
            raise ValueError("Detector has not been fitted. Call fit() first.")

        # Process each window and collect results
        all_seeds = []
        for proj in self.window_projections_:
            seeds = self.transform_kernel(proj)
            all_seeds.append(seeds)

        # Combine all seeds and remove duplicates
        if all_seeds:
            combined_seeds = pd.concat(all_seeds, ignore_index=True)
            # Remove duplicates by rounding to integer coordinates
            combined_seeds = combined_seeds.round().drop_duplicates()
            return combined_seeds
        else:
            # Return empty DataFrame with correct columns if no seeds found
            return pd.DataFrame(columns=self.core_axes)

    @abstractmethod
    def transform_kernel(self, X: xr.DataArray) -> pd.DataFrame:
        """
        Detect cells by combining detections from all windows.
        """
        pass

    def fit_transform(self, X: xr.DataArray, y: Any = None) -> pd.DataFrame:
        """
        Fit the detector and detect cells in one operation.

        Parameters
        ----------
        X : xr.DataArray
            Input data array.
        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        pd.DataFrame
            DataFrame containing detected cell positions and properties.
        """
        return self.fit(X).transform(X)
