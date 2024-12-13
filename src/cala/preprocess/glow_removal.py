from sklearn.base import BaseEstimator, TransformerMixin
import xarray as xr


class GlowRemover(BaseEstimator, TransformerMixin):
    def __init__(self, iter_axis: str = "frame"):
        self.base_brightness: float = 0
        self._iter_axis = iter_axis

    def fit(self, X, y=None):
        self.base_brightness = X.min(self._iter_axis).compute()
        return self

    def transform(self, X: xr.DataArray, y=None):
        return X - self.base_brightness
