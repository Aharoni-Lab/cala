from dataclasses import dataclass, field
from typing import List
from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BaseDemixer(BaseEstimator, TransformerMixin, ABC):
    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    iter_axis: str = "frames"
    spatial_axis: str = "spatial"

    @abstractmethod
    def fit_kernel(self, X: xr.DataArray, y):
        pass

    @abstractmethod
    def fit(self, X: xr.DataArray, y):
        pass

    @abstractmethod
    def transform_kernel(self, X: xr.DataArray, y):
        pass

    @abstractmethod
    def transform(self, X: xr.DataArray, y):
        pass
