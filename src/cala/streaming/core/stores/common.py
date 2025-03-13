from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set, Hashable, overload, Dict

import numpy as np
import xarray as xr

from .bodega import BodegaStore


# component_type = find_intersection_type_of(
#     base_type=FluorescentObject, instance=value
# )
# types = [component_type.__name__] * value.sizes[self.component_axis]


@dataclass(kw_only=True)
class CommonStore(BodegaStore):

    type_coord: str = "type_coord"

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
        empty_types = []
        warehouse = self.generate_warehouse(data_array=empty_array)
        warehouse = self.register_ids(warehouse, ids=empty_ids)
        self._warehouse = self.register_types(warehouse, empty_types)

    def register_types(self, warehouse: xr.DataArray, types: List[Hashable]):
        return warehouse.assign_coords(
            {
                self.type_coord: ([self.component_dim], types),
            }
        )

    def _validate_coords(self, coords: Set[Hashable]) -> None:
        if not (
            {self.id_coord, self.type_coord, self.component_dim}.issuperset(coords)
            and len(coords) > 1
        ):
            raise ValueError(
                f"The coordinates do not match the store structure.\n"
                f"\tProvided: {coords}\n"
                f"\tRequired: {self.id_coord, self.type_coord, self.component_dim}"
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

    def slice(
        self,
        ids: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """xarray "sel" method does not support vectorized selection for 2d coordinates (i.e. id=[1, 2, 3]), and thus
        we're selecting by type first and then multiple id's to leverage as much vectorization as possible.

        Args:
            ids:
            types:

        Returns:

        """
        if not ids:
            ids = self.types_to_ids(types)
        else:
            if not types:
                types = self._types
            ids = list(set(ids) & set(self.types_to_ids(types)))

        return (
            self._warehouse.set_xindex(self.id_coord)
            .sel({self.id_coord: ids})
            .reset_index(self.id_coord)
        )

    @overload
    def delete(
        self, ids: List[str], types: List[str], inplace: bool = False
    ) -> xr.DataArray: ...

    @overload
    def delete(
        self, ids: List[str], types: List[str], inplace: bool = True
    ) -> None: ...

    def delete(
        self,
        ids: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> Optional[xr.DataArray]:
        """xarray's `drop_sel` does not seem to support 2d coordinates yet,
        which is odd cause the`sel` method does support it.

        for now, we resort to a makeshift method where we select everything except the ones to drop.

        with a 2d coordinate support from drop_sel, this method would be:
        ```
        args = {self.id_coordinate: ids, self.type_coordinate: types}
        coords = {coord: idx for coord, idx in args.items() if idx is not None}

        if inplace:
            self._warehouse = self._warehouse.drop_sel(coords)
            return None
        else:
            return self._warehouse.drop_sel(coords)
        ```
        """
        if ids is None:
            ids = self.types_to_ids(types)

        elif ids and types:
            ids = list(set(ids) & set(self.types_to_ids(types)))

        result = self._warehouse.set_index(
            {self.component_dim: self.id_coord}
        ).drop_sel({self.component_dim: ids})

        if inplace:
            self._warehouse = result.assign_coords(
                {
                    self.id_coord: (
                        self.component_dim,
                        result.coords[self.component_dim].values.tolist(),
                    )
                }
            )
        else:
            return result.assign_coords(
                {
                    self.id_coord: (
                        self.component_dim,
                        result.coords[self.component_dim].values.tolist(),
                    )
                }
            )

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

    @abstractmethod
    def temporal_update(
        self, last_streamed_data: xr.DataArray, ids: List[str]
    ) -> None: ...

    """
    updates from new frames look different for each store.
    e.g. if you have frame_axis or its derivative, the store shape changes.
    otherwise the shape stays the same.

    an abstractmethod might not be good for this since parameters might be all different.
    """

    @property
    def _types(self) -> List[str]:
        return self._warehouse.coords[self.type_coord].values.tolist()

    @property
    def _ids(self) -> List[str]:
        return self._warehouse.coords[self.id_coord].values.tolist()

    @property
    def id_to_type(self) -> Dict[str, str]:
        return {id_: type_ for id_, type_ in zip(self._ids, self._types)}

    @property
    def type_to_ids(self) -> Dict[str, List[str]]:
        dict_ = defaultdict(list)
        for type_, id_ in zip(self._types, self._ids):
            dict_[type_].append(id_)
        return dict_

    def types_to_ids(self, types: List[str]) -> List[str]:
        return [id_ for type_ in types for id_ in self.type_to_ids[type_]]
