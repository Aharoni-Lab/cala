from dataclasses import dataclass

import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class GlowRemover(BaseEstimator, TransformerMixin):
    iter_axis: str = "frames"
    base_brightness_: float = None

    def fit(self, X, y=None):
        self.base_brightness_ = X.min(self.iter_axis).compute()
        return self

    def transform(self, X: xr.DataArray, y=None):
        return X - self.base_brightness_
