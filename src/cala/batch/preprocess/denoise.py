from dataclasses import dataclass, field
from typing import Literal, Self

import cv2
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Denoiser(BaseEstimator, TransformerMixin):
    # core_axes: The axes the filter convolves on. Defaults to ["height", "width"]
    core_axes: list[str] = field(default_factory=lambda: ["width", "height"])
    # method: One of "gaussian", "median", "bilateral". Defaults to "median".
    method: Literal["gaussian", "median", "bilateral"] = "gaussian"
    # kwargs: keyword args corresponding to the denoise method
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        methods = {
            "gaussian": cv2.GaussianBlur,
            "median": cv2.medianBlur,
            "bilateral": cv2.bilateralFilter,
        }
        if self.method not in methods:
            raise ValueError(
                f"denoise method '{self.method}' not understood. "
                f"Available methods are: {', '.join(methods.keys())}"
            )
        self.func = methods[self.method]

    def fit(self, X: xr.DataArray, y: None = None) -> Self:
        return self

    def transform(self, X: xr.DataArray, y: None = None) -> xr.DataArray:
        res = xr.apply_ufunc(
            self.func,
            X.astype(np.float32),
            input_core_dims=[self.core_axes],
            output_core_dims=[self.core_axes],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[X.dtype],
            kwargs=self.kwargs,
        )
        res = res.astype(X.dtype)
        return res.rename(f"{X.name}_denoised" if X.name else "denoised")
