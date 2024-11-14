from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple
import xarray as xr

from .estimation import select_base_frame, estimate_motion
from .transformation import apply_transform


class MotionCorrector(BaseEstimator, TransformerMixin):
    """Transformer that performs motion correction on video data.
    The MotionCorrector estimates motion parameters during fitting and applies
    the correction during transformation.
    """

    def __init__(
        self,
        dim: str = "frame",
    ):
        """
        Initializes the MotionCorrector transformer.
        Args:
        """
        self.motion_ = None
        self.base_frame = None
        self.dim_ = dim

    def fit(self, X: xr.DataArray, y=None):
        """Estimates motion parameters from the input data.
        Args:
            X (xr.DataArray): The input movie data.
            y: Ignored.
        Returns:
            MotionCorrector: The fitted transformer.
        """

        self.base_frame = select_base_frame(X, base_index=0)

        self.motion_ = estimate_motion(
            frames=X,
            base_frame=self.base_frame,
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
        # Ensure that fit has been called
        if not hasattr(self, "motion_"):
            raise RuntimeError("You must call 'fit' before 'transform'.")

        # Apply the estimated motion to correct the data
        corrected_X = apply_transform(
            varr=X,
            trans=self.motion_,
            fill=0,
            mesh_size=self.mesh_size,
        )
        return corrected_X
