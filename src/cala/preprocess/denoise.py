from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import xarray as xr


class Denoiser(BaseEstimator, TransformerMixin):
    methods = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    def __init__(self, core_axes: List[str], method: str = "median", **kwargs):
        """

        Args:
            method: One of "gaussian", "median", "bilateral". Defaults to "median".
            core_axes: The axes the filter convolves on. Defaults to ["height", "width"]
            **kwargs: Extra arguments for the filters. Check cv2 documentations for more details.
        """
        if method not in self.methods:
            raise ValueError(
                f"denoise method '{method}' not understood. "
                f"Available methods are: {', '.join(self.methods.keys())}"
            )
        self.method = method
        self.func = self.methods[method]
        self.core_axes = core_axes if core_axes is not None else ["height", "width"]
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        res = xr.apply_ufunc(
            self.func,
            X,
            input_core_dims=[self.core_axes],
            output_core_dims=[self.core_axes],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
            kwargs=self.kwargs,
        )
        res = res.astype(X.dtype)
        return res.rename(f"{X.name}_denoised" if X.name else "denoised")
