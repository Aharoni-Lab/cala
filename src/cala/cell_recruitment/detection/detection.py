from dataclasses import dataclass, field
from typing import Literal, List

import cv2
import numpy as np
import xarray as xr
from skimage.morphology import disk
from sklearn.base import BaseEstimator, TransformerMixin

from signal_processing import local_extreme


@dataclass
class Detector(BaseEstimator, TransformerMixin):
    """
    Args:
    chunk_size : int, optional
        Number of frames in each chunk, for which a max projection will be
        calculated. Default: 500
    method : str, optional
        Either `"rolling"` or `"random"`. Controls whether to use rolling window
        or random sampling of frames to construct chunks. Default: "rolling"
    step_size : int, optional
        Number of frames between the center of each chunk when stepping through
        the data with rolling windows. Only used if `method is "rolling"`. Default: 200
    num_chunks : int, optional
        Number of chunks to sample randomly. Only used if `method is "random"`.
        Default: 100
    local_max_radius : int, optional
        Radius (in pixels) of the disk window used for computing local maxima.
        Local maxima are defined as pixels with maximum intensity in such a
        window. Default: 10
    intensity_threshold : int, optional
        Intensity threshold for the difference between local maxima and its
        neighbours. Any local maxima that is not brighter than its neighbor
        (defined by the same disk window) by `intensity_threshold` intensity
        values will be filtered out. Default: 2
    """

    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    iter_axis: str = "frames"
    chunk_size: int = 500
    method: Literal["rolling", "random"] = "rolling"
    step_size: int = 200
    num_chunks: int = 100
    local_max_radius: int = 10
    intensity_threshold: int = 2
    max_projection_: xr.DataArray = field(default=None)

    def __post_init__(self):
        if self.method not in ["rolling", "random"]:
            raise ValueError("Method must be either 'rolling' or 'random'")

    def fit(self, X: xr.DataArray, y=None):
        """
        Fit Detector to have a statistical summary image of a set of frames.

        Parameters
        ----------
        X : xr.DataArray
            Input movie data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        self.max_projection_ = self._compute_max_projections(X)
        return self

    def transform(self, X):
        """
        Return the computed seeds.

        Parameters
        ----------
        X : xr.DataArray
            Input movie data.

        Returns
        -------
        seeds : pd.DataFrame
            DataFrame containing seeds.
        """
        if getattr(self, "max_projection_") is None:
            raise ValueError("Fit method must be run before transform.")

        local_maxima = xr.apply_ufunc(
            self._find_local_maxima,
            self.max_projection_,
            input_core_dims=[self.core_axes],
            output_core_dims=[self.core_axes],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.uint8],
            kwargs={
                "k0": 2,
                "k1": self.local_max_radius,
                "intensity_threshold": self.intensity_threshold,
            },
        ).sum("sample")

        seeds = (
            local_maxima.where(local_maxima > 0)
            .rename("seeds")
            .to_dataframe()
            .dropna()
            .reset_index()
        )
        return seeds[self.core_axes + ["seeds"]]

    def _compute_max_projections(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Compute the maximum intensity projections over specified chunks of frames.

        Parameters
        ----------
        data_array : xr.DataArray
            Input movie data.

        Returns
        -------
        max_projections : xr.DataArray
            Concatenated max projections over the sampled chunks.
        """
        frame_indices = data_array.coords[self.iter_axis]
        num_frames = len(frame_indices)

        max_indices = self._get_max_indices(num_frames)

        # Use a generator to process max projections lazily
        max_projections_gen = (
            data_array.isel({self.iter_axis: indices}).max(self.iter_axis)
            for indices in max_indices
        )

        max_projections = xr.concat(max_projections_gen, dim="sample")

        return max_projections

    def _get_max_indices(self, num_frames: int):
        if self.method == "rolling":
            num_steps = max(
                int(np.ceil((num_frames - self.chunk_size) / self.step_size)) + 1, 1
            )
            start_indices = (np.arange(num_steps) * self.step_size).astype(int)
            max_indices = [
                slice(start, min(start + self.chunk_size, num_frames))
                for start in start_indices
            ]
        elif self.method == "random":
            max_indices = [
                np.random.choice(
                    num_frames, size=min(self.chunk_size, num_frames), replace=False
                )
                for _ in range(self.num_chunks)
            ]
        else:
            raise ValueError("Method should be 'rolling' or 'random'")

        return max_indices

    @staticmethod
    def _find_local_maxima(
        frame: np.ndarray, k0: int, k1: int, intensity_threshold: int | float
    ) -> np.ndarray:
        """
        Compute local maxima of a frame using a range of kernel sizes.

        Parameters
        ----------
        frame : np.ndarray
            The input frame.
        k0 : int
            The lower bound (inclusive) of the range of kernel sizes.
        k1 : int
            The upper bound (exclusive) of the range of kernel sizes.
        intensity_threshold : Union[int, float]
            Intensity threshold for the difference between local maxima and its neighbours.

        Returns
        -------
        local_maxima : np.ndarray
            Binary image of local maxima, with 1 at local maxima positions.
        """
        max_result = np.zeros_like(frame, dtype=np.uint8)

        for kernel_size in range(k0, k1):
            structuring_element = disk(kernel_size)
            frame_max = local_extreme(
                frame, structuring_element, intensity_threshold, mode="max"
            )
            np.logical_or(max_result, frame_max.astype(np.uint8), out=max_result)

        num_labels, labeled_maxima = cv2.connectedComponents(max_result)

        local_maxima = np.zeros_like(max_result, dtype=np.uint8)

        for label in range(1, num_labels):
            coords = np.argwhere(labeled_maxima == label)

            if coords.shape[0] > 1:
                median_coord = np.median(coords, axis=0).astype(int)
                local_maxima[tuple(median_coord)] = 1
            else:
                local_maxima[tuple(coords[0])] = 1

        return local_maxima
