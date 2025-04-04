from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class GlowRemover(BaseEstimator, TransformerMixin):
    iter_axis: str = "frame"
    base_brightness_: xr.DataArray = field(init=False)

    def fit(self, X: xr.DataArray, y: None = None) -> Self:
        self.base_brightness_ = X.min(self.iter_axis).compute()
        return self

    def transform(self, X: xr.DataArray, y: None = None) -> xr.DataArray:
        return X - self.base_brightness_
