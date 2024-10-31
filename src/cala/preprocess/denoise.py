from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import xarray as xr


class Denoiser(BaseEstimator, TransformerMixin):
    methods = {
        "gaussian": cv2.GaussianBlur,
        "median": cv2.medianBlur,
        "bilateral": cv2.bilateralFilter,
    }

    def __init__(self, method: str, **kwargs):
        if method not in self.methods:
            raise ValueError(
                f"denoise method '{method}' not understood. "
                f"Available methods are: {', '.join(self.methods.keys())}"
            )
        self.method = method
        self.func = self.methods[method]
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        res = xr.apply_ufunc(
            self.func,
            X,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
            kwargs=self.kwargs,
        )
        res = res.astype(X.dtype)
        return res.rename(X.name + "_denoised")
