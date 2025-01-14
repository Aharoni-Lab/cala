from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Self, ClassVar

from xarray import DataArray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


def track_calls(method):
    """A decorator that sets an instance attribute whenever `method` is called."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Mark that this instance has been 'fit'
        self._has_been_fitted = True
        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    # core_axes: the axes in which filters will be applied against
    core_axes: List[str] = field(default_factory=lambda: ["width", "height"])
    # iter_axis: the axis in which the filtering will be parallelized against
    iter_axis: str = "frames"
    # spatial_axis: the multiplexed axis that encompasses the entire visual space of the movie (width x height, radial x angular, etc.)
    spatial_axis: str = "spatial"
    # reusing_fit: True if transform is being applied on a different dataset from the one used in fit.
    reusing_fit: bool = True
    _stateless: ClassVar[bool] = False
    _has_been_fitted: bool = False

    def _validate_axes(self, X: DataArray) -> None:
        """Validate that required axes exist in the DataArray."""
        missing_axes = []

        # Check core axes
        for axis in self.core_axes:
            if axis not in X.dims:
                missing_axes.append(axis)

        # Check iteration axis
        if self.iter_axis not in X.dims:
            missing_axes.append(self.iter_axis)

        if missing_axes:
            raise ValueError(
                f"DataArray is missing dimensions: {missing_axes}. "
                f"Available dimensions are: {list(X.dims)}"
            )

    @abstractmethod
    def fit_kernel(self, X: DataArray, seeds: DataFrame):
        pass

    @abstractmethod
    def transform_kernel(self, X: DataArray, seeds: DataFrame):
        pass

    @track_calls
    def fit(self, X: DataArray, y: DataFrame, **fit_params: dict) -> Self:
        self._validate_axes(X)
        self.fit_transform_shared_preprocessing(X=X, seeds=y)
        self.fit_kernel(X=X, seeds=y)
        return self

    def transform(self, X: DataArray, y: DataFrame):
        self._validate_axes(X)

        if ~(self._has_been_fitted or self._stateless):
            raise NotFittedError("The filter has not been fitted.")

        elif self.reusing_fit:
            X, y = self.fit_transform_shared_preprocessing(X=X, seeds=y)

        return self.transform_kernel(X=X, seeds=y)

    def fit_transform(self, X: DataArray, y: DataFrame = None, **fit_params: dict):
        return self.fit(X, y, **fit_params).transform(X, y)

    @abstractmethod
    def fit_transform_shared_preprocessing(self, X: DataArray, seeds: DataFrame):
        pass
