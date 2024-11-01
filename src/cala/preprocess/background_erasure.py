from typing import Literal, List
import cv2
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from sklearn.base import BaseEstimator, TransformerMixin


class BackgroundEraser(BaseEstimator, TransformerMixin):
    """Transformer that removes background from video frames using specified methods.

    The BackgroundEraser applies background removal techniques to each frame of a video.
    Two methods are available:

    - 'uniform': Background is estimated by convolving each frame with a uniform/mean kernel
      and then subtracting it from the frame.
    - 'tophat': Applies a morphological tophat operation to each frame using a disk-shaped kernel.

    Attributes:
        method (str): The method used to remove the background.
        kernel_size (int): Size of the kernel used for background removal.
    """

    def __init__(
        self,
        core_axes: List[str, ...] = None,
        method: Literal["uniform", "tophat"] = "uniform",
        kernel_size: int = 3,
    ):
        """Initializes the BackgroundEraser transformer.

        Args:
            core_axes (List[str, ...], optional): The core dimensions of the video, or the
                dimensions on which the filter convolves on. Defaults to ["height", "width"].
            method (Literal["uniform", "tophat"], optional): The method used to remove the background.
                Should be either "uniform" or "tophat". Defaults to "uniform".
            kernel_size (int, optional): Window size of kernels used for background removal,
                specified in pixels. If method == "uniform", this will be the size of a box kernel
                convolved with each frame. If method == "tophat", this will be the radius of a
                disk kernel used for morphological operations. Defaults to 3.
        """
        self.core_axes = core_axes if core_axes is not None else ["height", "width"]
        self.method = method
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        """Fits the transformer to the data.

        This transformer does not learn from the data, so this method simply returns self.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            BackgroundEraser: The fitted transformer.
        """
        return self

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
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
        res.name = f"{X.name}_subtracted" if X.name else "background_subtracted"
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
            kernel = self.disk(self.kernel_size)
            return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
        else:
            raise ValueError("Method must be either 'uniform' or 'tophat'.")

    @staticmethod
    def disk(radius: int) -> np.ndarray:
        """Creates a disk-shaped structuring element with the given radius.

        Args:
            radius (int): Radius of the disk.

        Returns:
            np.ndarray: 2D binary array with ones inside the disk and zeros outside.

        Raises:
            ValueError: If radius is not a non-negative integer.
        """
        if radius < 0 or not isinstance(radius, int):
            raise ValueError("Radius must be a non-negative integer.")

        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = x**2 + y**2 <= radius**2
        kernel = mask.astype(np.uint8)
        return kernel
