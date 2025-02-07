from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BaseMotionCorrector(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for parallel motion corrector.
    """

    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    iter_axis: str = "frames"
    anchor_frame_index: int = None
    anchor_frame_: xr.DataArray = None
    motion_: xr.DataArray = None

    def _fit_kernel(
        self,
        anchor_frame: xr.DataArray,
        current_frame: xr.DataArray,
    ):
        """
        Define the fit ufunc to apply. Can be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _transform_kernel(self, frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """
        Define the transform ufunc to apply. Must be implemented by subclasses.
        """
        pass

    def fit(self, X: xr.DataArray, y=None, **fit_kwargs):
        """
        Fit method. For transformers that don't need fitting, simply return self.
        Subclasses can override this if fitting is required.
        """
        if self.anchor_frame_index is None:
            raise ValueError(
                "Calculating optimal anchor frame has not been implemented yet. anchor_frame_index is required."
            )
        elif self.anchor_frame_ is None:
            self.anchor_frame_ = self.anchor_by_index(
                X, anchor_index=self.anchor_frame_index
            )
        self.motion_ = xr.apply_ufunc(
            self._fit_kernel,
            X,
            input_core_dims=[self.core_axes],
            output_core_dims=[["shift_dim"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
            kwargs=fit_kwargs,
        )

        return self

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """
        Apply the _transform_kernel in parallel using xarray.
        """
        if self.motion_ is None:
            raise ValueError(
                "Motion has not been calculated yet. Fit method must be run before transform."
            )

        return xr.apply_ufunc(
            self._transform_kernel,
            X,
            self.motion_,
            input_core_dims=[self.core_axes, ["shift_dim"]],
            output_core_dims=[self.core_axes],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
        )

    def anchor_by_index(
        self, video: xr.DataArray, anchor_index: Optional[int] = 0
    ) -> xr.DataArray:
        """
        Select an anchor frame from the video frames.

        Parameters:
        - frames: xarray.DataArray containing video frames.
        - anchor_index: Index of the anchor frame.

        Returns:
        - xarray.DataArray of the anchor frame.
        """
        if anchor_index < 0 or anchor_index >= video.sizes[self.iter_axis]:
            raise IndexError("anchor_index is out of bounds.")
        anchor_frame = video.isel({self.iter_axis: anchor_index})
        return xr.DataArray(
            anchor_frame,
            dims=self.core_axes,
            attrs={
                "description": "Anchor frame extracted from video",
                "anchor_index": anchor_index,
            },
        )
