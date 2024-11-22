from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    iter_axis: str = "frames"

    @abstractmethod
    def fit_kernel(self, X):
        pass

    @abstractmethod
    def transform_kernel(self, X, y):
        pass

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass
