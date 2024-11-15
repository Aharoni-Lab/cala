from typing import List, Tuple, Optional

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin

from .transformation import RigidTranslation
from .transformation.base import Transformation


class MotionCorrector(BaseEstimator, TransformerMixin):
    """Transformer that performs motion correction on video data.
    The MotionCorrector estimates motion parameters during fitting and applies
    the correction during transformation.
    """

    def __init__(
        self,
        core_axes: List[str] = None,
        iter_axis: str = "frames",
        base_frame_index: Optional[int] = None,
        transformations: Optional[List[Transformation]] = None,
    ):
        """
        Initializes the MotionCorrector transformer.
        Args:
        """
        self.core_axes = core_axes if core_axes is not None else ["height", "width"]
        self.iter_axis = iter_axis
        self.base_frame_index = base_frame_index
        self.transformations = (
            transformations if transformations is not None else [RigidTranslation()]
        )
        self.base_frame_ = None
        self.motion_ = None

    def fit(self, X: xr.DataArray, y=None):
        """Estimates motion parameters from the input data.
        Args:
            X (xr.DataArray): The input movie data.
            y: Ignored.
        Returns:
            MotionCorrector: The fitted transformer.
        """

        if self.base_frame_index is None:
            raise ValueError(
                "Calculating optimal base frame has not been implemented yet. base_frame_index is required."
            )
        else:
            self.base_frame_ = self.select_base_frame(
                X, base_index=self.base_frame_index
            )

        self.motion_ = self.estimate_motion(
            video=X,
        )
        return self

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        """Applies motion correction to the input data using the estimated parameters.
        Args:
            X (xr.DataArray): The input movie data.
            y: Ignored.
        Returns:
            xr.DataArray: The motion-corrected movie data.
        Raises:
            RuntimeError: If the transformer has not been fitted yet.
        """
        if not hasattr(self, "motion_"):
            raise RuntimeError("You must call 'fit' before 'transform'.")

        corrected_frames = X

        for transformation in self.transformations:
            transform_name = transformation.__class__.__name__.lower()
            params = self.motion_[transform_name]

            corrected_frames = xr.apply_ufunc(
                transformation.apply_transformation,
                X,
                params,
                input_core_dims=[
                    self.core_axes,
                    [],
                ],  # Core dimensions for frame and params
                output_core_dims=[self.core_axes],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[X.dtype],
            )

        corrected_frames.attrs["description"] = (
            "Stabilized video frames with motion corrections applied."
        )

        return corrected_frames

    def select_base_frame(
        self, video: xr.DataArray, base_index: Optional[int] = 0
    ) -> xr.DataArray:
        """
        Select a base frame from the video frames.

        Parameters:
        - frames: xarray.DataArray containing video frames.
        - base_index: Index of the base frame.

        Returns:
        - xarray.DataArray of the base frame.
        """
        if base_index < 0 or base_index >= video.sizes[self.iter_axis]:
            raise IndexError("base_index is out of bounds.")

        base_frame = video.isel({self.iter_axis: base_index}).compute().astype(np.uint8)
        return xr.DataArray(
            base_frame,
            dims=self.core_axes,
            attrs={
                "description": "Base frame extracted from video",
                "base_index": base_index,
            },
        )

    def estimate_motion(
        self,
        video: xr.DataArray,
    ) -> xr.Dataset:
        """
        Estimate transformations for each frame relative to the base frame.

        Parameters:
        - video: xarray.DataArray containing video frames.

        Returns:
        - xarray.Dataset containing transformation parameters for each frame.
        """
        motion_dict = {}
        for transformation in self.transformations:
            transform_name = transformation.__class__.__name__.lower()
            transform_params = self.estimate_transformation(
                transformation=transformation,
                video=video,
            )
            motion_dict[transform_name] = transform_params

        return xr.Dataset(motion_dict, coords={self.iter_axis: video[self.iter_axis]})

    def estimate_transformation(
        self,
        transformation: Transformation,
        video: xr.DataArray,
    ) -> xr.Dataset:
        """
        Estimate a specific transformation for each frame relative to the base frame.

        Parameters:
        - transformation: An instance of a Transformation subclass.
        - video: xarray.DataArray containing video frames.

        Returns:
        - xarray.Dataset containing transformation parameters for each frame.
        """

        # Obtain a sample parameter dict to extract parameter names
        sample_params = transformation.compute_shift(
            self.base_frame_.values, self.base_frame_.values
        )
        param_names = list(sample_params.keys())

        def compute_params(current_frame: np.ndarray) -> Tuple[float, ...]:
            """Compute transformation parameters for a single frame."""
            params = transformation.compute_shift(
                self.base_frame_.values, current_frame
            )
            return tuple(params[name] for name in param_names)

        params = xr.apply_ufunc(
            compute_params,
            video,
            input_core_dims=[self.core_axes],
            output_core_dims=[[] for _ in param_names],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float for _ in param_names],
        )

        data_vars = {param: params[i] for i, param in enumerate(param_names)}

        params_dataset = xr.Dataset(
            data_vars=data_vars,
            coords={self.iter_axis: video[self.iter_axis]},
            attrs={
                "description": f"Transformation parameters for {transformation.__class__.__name__}."
            },
        )

        return params_dataset
