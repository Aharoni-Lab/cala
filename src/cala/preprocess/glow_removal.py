from sklearn.base import BaseEstimator, TransformerMixin
import xarray as xr


class GlowRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, iter_axis: str = "frame"):
        self._iter_axis = iter_axis

    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray, y=None):
        base_brightness = X.min(self._iter_axis).compute()
        return X - base_brightness
