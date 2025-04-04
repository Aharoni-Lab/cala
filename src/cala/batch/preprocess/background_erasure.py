from dataclasses import dataclass, field
from typing import Literal, Self

import cv2
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from skimage.morphology import disk
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BackgroundEraser(BaseEstimator, TransformerMixin):
    """Transformer that removes background from video frames using specified methods."""

    # - 'uniform': Background is estimated by convolving each frame with a uniform/mean kernel
    # and then subtracting it from the frame.
    # - 'tophat': Applies a morphological tophat operation to each frame using a disk-shaped kernel.
    core_axes: list[str] = field(default_factory=lambda: ["height", "width"])
    # method (str): The method used to remove the background.
    method: Literal["uniform", "tophat"] = "uniform"
    # kernel_size (int): Size of the kernel used for background removal.
    kernel_size: int = 3

    def fit(self, X: xr.DataArray, y: None = None) -> Self:
        """Fits the transformer to the data.

        This transformer does not learn from the data, so this method simply returns self.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            BackgroundEraser: The fitted transformer.
        """
        return self

    def transform(self, X: xr.DataArray, y: None = None) -> xr.DataArray:
        """Removes background from a video.

        This function removes background frame by frame. Two methods are available:

        - If method == "uniform", the background is estimated by convolving the frame with a
          uniform/mean kernel and then subtracting it from the frame.
        - If method == "tophat", a morphological tophat operation is applied to each frame.

        Args:
            X (xr.DataArray): The input video data, should have dimensions "frame", "height",
                and "width".
            y: Ignored. Not used, present for API consistency by convention.

        Returns:
            xr.DataArray: The resulting video with background removed. Same shape as input `X`
            but will have "_subtracted" appended to its name.

        Raises:
            ValueError: If input DataArray does not have the required dimensions.

        See Also:
            Morphological operations in OpenCV:
            https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html
        """

        # Apply the filter per frame using xarray's apply_ufunc
        res = xr.apply_ufunc(
            self.apply_filter_per_frame,
            X,
            input_core_dims=[self.core_axes],
            output_core_dims=[self.core_axes],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
        )
        res = res.astype(X.dtype)
        res.name = f"{X.name}_subtracted" if X.name != "None" else "background_subtracted"
        return res

    def apply_filter_per_frame(self, frame: np.ndarray) -> np.ndarray:
        """Removes background from a single frame.

        Args:
            frame (np.ndarray): The input frame.

        Returns:
            np.ndarray: The frame with background removed.

        Raises:
            ValueError: If an invalid method is specified.
        """
        if self.method == "uniform":
            background = uniform_filter(frame, size=self.kernel_size)
            return frame - background
        elif self.method == "tophat":
            kernel = disk(self.kernel_size)
            return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
        else:
            raise ValueError("Method must be either 'uniform' or 'tophat'.")
