from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Set, Hashable, overload, Dict

import numpy as np
import xarray as xr


@dataclass(kw_only=True)
class BaseStore(ABC):
    dimensions: Tuple[str, ...]
    component_dim: str

    _id_coord: str = "id_coord"
    _type_coord: str = "type_coord"

    _warehouse: xr.DataArray = field(default_factory=lambda: xr.DataArray())

    def __post_init__(self):
        if self.component_dim not in self.dimensions:
            raise ValueError(
                f"component_axis {self.component_dim} must be in dims {self.dimensions}."
            )
        empty_array = np.empty(shape=tuple([0] * len(self.dimensions)))
        empty_ids = []
        empty_types = []
        self.generate_warehouse(
            data_array=empty_array, ids=empty_ids, types=empty_types
        )

    def generate_warehouse(
        self, data_array: np.ndarray | xr.DataArray, ids: List[str], types: List[str]
    ) -> xr.DataArray:
        return xr.DataArray(
            data_array,
            dims=self.dimensions,
            coords={
                self._type_coord: ([self.component_dim], types),
                self._id_coord: ([self.component_dim], ids),
            },
        )

    @property
    def warehouse(self) -> xr.DataArray:
        return self._warehouse

    @warehouse.setter
    def warehouse(self, value: xr.DataArray):
        if self._validate_dims(value.dims) and self._validate_coords(
            set(value.coords.keys())
        ):
            self._warehouse = value

        else:
            raise ValueError(
                f"Inappropriate storage dimensions. Values in this storage must have:\n\tdimensions: {self.dimensions}\n\tcoordinates:{self._id_coord, self._type_coord} that are attached to {self.component_dim}.\nRefer to generate_warehouse method for formatting an unlabeled array."
            )

    def _validate_dims(self, dimensions: Tuple[Hashable, ...]) -> bool:
        return set(self.dimensions) == set(dimensions)

    def _validate_coords(self, coords: Set[Hashable]) -> bool:
        """this also validates the coordinate attachment to index.

        Args:
            coords:

        Returns:

        """
        return {self._id_coord, self._type_coord, self.component_dim}.issuperset(coords)

    @overload
    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[str],
        inplace=True,
    ) -> None: ...

    @overload
    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[str],
        inplace=False,
    ) -> xr.DataArray: ...

    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[str],
        inplace=False,
    ) -> Optional[xr.DataArray]:
        # make sure the data_array has no id / type coordinates
        to_insert = self.generate_warehouse(data_array, ids, types)
        if inplace:
            if not self.warehouse.dims:  # not sure about allowing this.
                self._warehouse = to_insert
                return None
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
            self._warehouse.set_xindex(self._id_coord)
            .sel({self._id_coord: ids})
            .reset_index(self._id_coord)
        )

    def where(self, condition, other: Any, drop: bool = False) -> xr.DataArray:
        """refer to xarray dataarray method for docs
        other: function or value to use for replacement
        drop: whether to drop the condition dimension"""
        return self._warehouse.where(condition, other=other, drop=drop)

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
            {self.component_dim: self._id_coord}
        ).drop_sel({self.component_dim: ids})

        if inplace:
            self._warehouse = result.assign_coords(
                {
                    self._id_coord: (
                        self.component_dim,
                        result.coords[self.component_dim].values.tolist(),
                    )
                }
            )
        else:
            return result.assign_coords(
                {
                    self._id_coord: (
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
        data_coords = data.coords[self._id_coord].values.tolist()

        if inplace:
            self._warehouse.set_xindex(self._id_coord).loc[
                {self._id_coord: data_coords}
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
        return self._warehouse.coords[self._type_coord].values.tolist()

    @property
    def _ids(self) -> List[str]:
        return self._warehouse.coords[self._id_coord].values.tolist()

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
