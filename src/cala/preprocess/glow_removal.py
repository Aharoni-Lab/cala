from sklearn.base import BaseEstimator, TransformerMixin
import xarray as xr


class GlowRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray, y=None):
        base_brightness = X.min("frame").compute()
        return X - base_brightness
