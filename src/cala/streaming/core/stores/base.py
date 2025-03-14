from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import Tuple, Hashable, Any, Type

import numpy as np
from xarray import DataArray


@dataclass(kw_only=True)
class BaseStore(ABC, metaclass=ABCMeta):
    dimensions: Tuple[str, ...]

    _warehouse: DataArray = field(default_factory=lambda: DataArray())

    def __post_init__(self):
        empty_array = np.empty(shape=tuple([0] * len(self.dimensions)))

        self._warehouse = self.generate_warehouse(data_array=empty_array)

    @property
    @abstractmethod
    def data_type(self):
        return type

    @property
    def warehouse(self) -> DataArray:
        return self._warehouse

    @warehouse.setter
    def warehouse(self, value: DataArray):
        self._validate_warehouse(value)
        self._warehouse = value

    def generate_warehouse(self, data_array: np.ndarray | DataArray) -> DataArray:
        return DataArray(
            data_array,
            dims=self.dimensions,
        )

    def _validate_warehouse(self, warehouse: DataArray) -> None:
        self._validate_dims(warehouse.dims)

    def _validate_dims(self, dimensions: Tuple[Hashable, ...]) -> None:
        if not set(self.dimensions) == set(dimensions):
            raise ValueError(
                "The dimensions do not match the store structure.\n"
                f"\tProvided: {dimensions}\n"
                f"\tRequired: {self.dimensions}"
            )

    def where(self, condition, other: Any, drop: bool = False) -> DataArray:
        """refer to xarray dataarray method for docs
        other: function or value to use for replacement
        drop: whether to drop the condition dimension"""
        return self._warehouse.where(condition, other=other, drop=drop)

    def get(self, type_: Type = None) -> DataArray:
        if issubclass(type_, self.data_type):
            return self.warehouse
        else:
            raise TypeError(
                f"Store {self.__class__.__name__}'s data type is {self.data_type}. Queried type: {type_}."
            )
