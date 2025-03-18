from abc import abstractmethod
from dataclasses import dataclass
from typing import Hashable, List, Set, overload, Optional
from uuid import uuid4, UUID

import numpy as np
import xarray as xr

from . import LowkeyStore


@dataclass(kw_only=True)
class MidkeyStore(LowkeyStore):
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
        self._warehouse = self._register_ids(warehouse, ids=empty_ids)

    def _is_unregistered(self, components: xr.DataArray) -> bool:
        return len([k for k, v in components.coords.items() if k == self.id_coord]) == 0

    def _generate_id_coord(self, value) -> List[UUID]:
        return [uuid4() for _ in range(value.sizes[self.component_axis])]

    def _register_ids(self, warehouse: xr.DataArray, ids: List[Hashable]):
        return warehouse.assign_coords(
            {
                self.id_coord: ([self.component_dim], ids),
            }
        )

    def _validate_warehouse(self, warehouse: xr.DataArray) -> None:
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

    def slice(
        self,
        ids: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """xarray "sel" method does not support vectorized selection for 2d coordinates (i.e. id=[1, 2, 3]), and thus
        we're selecting by type first and then multiple id's to leverage as much vectorization as possible.

        Args:
            ids:

        Returns:

        """
        return (
            self._warehouse.set_xindex(self.id_coord)
            .sel({self.id_coord: ids})
            .reset_index(self.id_coord)
        )

    @overload
    def insert(
        self,
        to_insert: xr.DataArray,
        inplace=True,
    ) -> None: ...

    @overload
    def insert(
        self,
        to_insert: xr.DataArray,
        inplace=False,
    ) -> xr.DataArray: ...

    def insert(
        self,
        to_insert: xr.DataArray,
        inplace=False,
    ) -> Optional[xr.DataArray]:
        """

        Args:
            to_insert: only accepts Xarray DataArray formatted for the store. Refer to generate_warehouse method.
            inplace:

        Returns:

        """

        if inplace and np.sum(self.warehouse.shape) == 0:
            self._warehouse = to_insert
            return None

        self._validate_dims(to_insert.dims)
        self._validate_coords(set(to_insert.coords.keys()))
        already_exist = set(to_insert.coords[self.id_coord].values.tolist()) & set(
            self._ids
        )
        if not already_exist == set():
            raise ValueError(
                f"IDs {already_exist} already exist in store. Cannot be inserted."
            )

        if inplace:
            self._warehouse = xr.concat(
                [self._warehouse, to_insert], dim=self.component_dim
            )
        else:
            return xr.concat([self._warehouse, to_insert], dim=self.component_dim)

    @property
    def _ids(self) -> List[str]:
        return self._warehouse.coords[self.id_coord].values.tolist()

    @overload
    def update(self, data: xr.DataArray, inplace: bool = True) -> None: ...

    @overload
    def update(self, data: xr.DataArray, inplace: bool = False) -> xr.DataArray: ...

    def update(
        self, data: xr.DataArray, inplace: bool = False
    ) -> Optional[xr.DataArray]:
        """only allows formatted dataarray with appropriate dims and coords. can use the generate_warehouse method beforehand."""
        data_coords = data.coords[self.id_coord].values.tolist()

        if inplace:
            self._warehouse.set_xindex(self.id_coord).loc[
                {self.id_coord: data_coords}
            ] = data  # shape safe
        else:
            return self.slice(data_coords)
