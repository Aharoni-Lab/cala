from abc import abstractmethod
from dataclasses import dataclass
from typing import Hashable, List, Set

import numpy as np
from xarray import DataArray

from . import BaseStore


# ids = [uuid4() for _ in range(value.sizes[self.component_axis])]


@dataclass(kw_only=True)
class BodegaStore(BaseStore):
    component_dim: str

    id_coord: str = "id_coord"

    @property
    @abstractmethod
    def data_type(self):
        return type

    def __post_init__(self):
        if self.component_dim not in self.dimensions:
            raise ValueError(
                f"component_axis {self.component_dim} must be in dims {self.dimensions}."
            )
        empty_array = np.empty(shape=tuple([0] * len(self.dimensions)))
        empty_ids = []
        warehouse = self.generate_warehouse(data_array=empty_array)
        self._warehouse = self.register_ids(warehouse, ids=empty_ids)

    def _is_unregistered(self, components: DataArray) -> bool:
        return len([k for k, v in components.coords.items() if k == self.id_coord]) == 0

    def register_ids(self, warehouse: DataArray, ids: List[Hashable]):
        return warehouse.assign_coords(
            {
                self.id_coord: ([self.component_dim], ids),
            }
        )

    def _validate_warehouse(self, warehouse: DataArray) -> None:
        self._validate_dims(warehouse.dims)
        self._validate_coords(set(warehouse.coords.keys()))

    def _validate_coords(self, coords: Set[Hashable]) -> None:
        if not (
            {self.id_coord, self.component_dim}.issuperset(coords) and len(coords) > 1
        ):
            raise ValueError(
                f"The coordinates do not match the store structure.\n"
                f"\tProvided: {coords}\n"
                f"\tRequired: {self.id_coord, self.component_dim}"
            )
