from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple
import xarray as xr

from .estimation import estimate_motion
from .transformation import apply_transform


class MotionCorrector(BaseEstimator, TransformerMixin):
    """Transformer that performs motion correction on video data.

    The MotionCorrector estimates motion parameters during fitting and applies
    the correction during transformation.

    Attributes:
        dim (str): The dimension along which motion estimation should be carried out.
        num_chunks (int): Number of frames/chunks to combine for the recursive algorithm.
        frames_per_chunk (int, optional): Number of frames in each parallel task.
        error_threshold (float): Error threshold between estimated shifts from two alternative methods.
        aggregation (str): How frames should be aggregated to generate the template for each chunk.
        upsample (int): The upsample factor for sub-pixel accuracy in motion estimation.
        circularity_threshold (float, optional): Circularity threshold to check frame suitability as template.
        qc_padding (int): Amount of zero padding when checking for frame quality.
        mesh_size (Tuple[int, int], optional): Number of control points for the BSpline mesh.
        n_iter (int): Max number of iterations for gradient descent in non-rigid motion estimation.
        bin_threshold (float, optional): Intensity threshold for binarizing frames in non-rigid estimation.
    """

    def __init__(
        self,
        dim: str = "frame",
        num_chunks: int = 3,
        frames_per_chunk: Optional[int] = None,
        error_threshold: float = 5.0,
        aggregation: str = "mean",
        upsample: int = 100,
        circularity_threshold: Optional[float] = None,
        qc_padding: int = 100,
        mesh_size: Optional[Tuple[int, int]] = None,
        n_iter: int = 100,
        bin_threshold: Optional[float] = None,
    ):
        """
        Initializes the MotionCorrector transformer.

        Args:
            dim (str, optional): The dimension along which motion estimation should be carried out.
                Defaults to 'frame'.
            num_chunks (int, optional): Number of frames/chunks to combine for the recursive algorithm.
                Defaults to 3.
            frames_per_chunk (int, optional): Number of frames in each parallel task.
                Defaults to None.
            error_threshold (float, optional): Error threshold between estimated shifts from two alternative methods.
                Defaults to 5.0.
            aggregation (str, optional): How frames should be aggregated to generate the template.
                Should be 'mean' or 'max'. Defaults to 'mean'.
            upsample (int, optional): The upsample factor for sub-pixel accuracy in motion estimation.
                Defaults to 100.
            circularity_threshold (float, optional): Circularity threshold to check frame suitability as template.
                Defaults to None.
            qc_padding (int, optional): Amount of zero padding when checking for frame quality.
                Defaults to 100.
            mesh_size (Tuple[int, int], optional): Number of control points for the BSpline mesh.
                Defaults to None.
            n_iter (int, optional): Max number of iterations for gradient descent in non-rigid motion estimation.
                Defaults to 100.
            bin_threshold (float, optional): Intensity threshold for binarizing frames in non-rigid estimation.
                Defaults to None.
        """
        self.motion_ = None
        self.dim = dim
        self.num_chunks = num_chunks
        self.frames_per_chunk = frames_per_chunk
        self.error_threshold = error_threshold
        self.aggregation = aggregation
        self.upsample = upsample
        self.circularity_threshold = circularity_threshold
        self.qc_padding = qc_padding
        self.mesh_size = mesh_size
        self.n_iter = n_iter
        self.bin_threshold = bin_threshold

    def fit(self, X: xr.DataArray, y=None):
        """Estimates motion parameters from the input data.

        Args:
            X (xr.DataArray): The input movie data.
            y: Ignored.

        Returns:
            MotionCorrector: The fitted transformer.
        """
        # Estimate motion parameters and store them
        self.motion_ = estimate_motion(
            varr=X,
            dim=self.dim,
            num_chunks=self.num_chunks,
            frames_per_chunk=self.frames_per_chunk,
            error_threshold=self.error_threshold,
            aggregation=self.aggregation,
            upsample=self.upsample,
            circularity_threshold=self.circularity_threshold,
            qc_padding=self.qc_padding,
            mesh_size=self.mesh_size,
            n_iter=self.n_iter,
            bin_threshold=self.bin_threshold,
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
