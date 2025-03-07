from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Type, Optional, Any, Tuple, Set, Hashable, overload

import numpy as np
import xarray as xr


@dataclass
class BaseStore(ABC):
    component_axis: str

    id_coord: str = "id_"
    type_coord: str = "type_"

    _warehouse: xr.DataArray = None

    @property
    @abstractmethod
    def dims(self) -> Tuple[str, ...]: ...

    def __post_init__(self):
        empty_array = np.empty(shape=tuple([0] * len(self.dims)))
        empty_ids = []
        empty_types = []
        self.generate_store(data_array=empty_array, ids=empty_ids, types=empty_types)

    def generate_store(
        self, data_array: np.ndarray | xr.DataArray, ids: List[str], types: List[Type]
    ) -> xr.DataArray:
        data = xr.DataArray(
            data_array,
            dims=self.dims,
            coords={
                self.type_coord: ([self.component_axis], types),
                self.id_coord: ([self.component_axis], ids),
            },
        )
        return data.set_index({self.component_axis: [self.type_coord, self.id_coord]})

    @property
    def warehouse(self) -> xr.DataArray:
        # Must output an array with id coordinates
        return self._warehouse

    @warehouse.setter
    def warehouse(self, value: xr.DataArray):
        if self._validate_dims(value.dims) and self._validate_coords(
            set(value.coords.keys())
        ):
            self._warehouse = value

        else:
            raise ValueError(
                f"Inappropriate storage dimensions. Values in this storage must have:\n\tdimensions: {self.dims}\n\tcoordinates:{self.id_coord, self.type_coord} that are attached to {self.component_axis}.\nRefer to generate_store method for formatting an unlabeled array."
            )

    def _validate_dims(self, dimensions: Tuple[Hashable, ...]) -> bool:
        return set(self.dims) == set(dimensions)

    def _validate_coords(self, coords: Set) -> bool:
        """this also validates the coordinate attachment to index.

        Args:
            coords:

        Returns:

        """
        return {self.id_coord, self.type_coord, self.component_axis} == coords

    @overload
    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[Type],
        inplace=True,
    ) -> None: ...

    @overload
    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[Type],
        inplace=False,
    ) -> xr.DataArray: ...

    def insert(
        self,
        data_array: np.ndarray | xr.DataArray,
        ids: List[str],
        types: List[Type],
        inplace=False,
    ) -> Optional[xr.DataArray]:
        # make sure the data_array has no id / type coordinates
        to_insert = self.generate_store(data_array, ids, types)
        if inplace:
            self._warehouse = xr.concat(
                [self._warehouse, to_insert], dim=self.component_axis
            )
        else:
            return xr.concat([self._warehouse, to_insert], dim=self.component_axis)

    def slice(
        self,
        ids: Optional[List[str]],
        types: Optional[List[Type]],
    ) -> xr.DataArray:
        return self._warehouse.sel({self.id_coord: ids, self.type_coord: types})

    def where(self, condition, other: Any, drop: bool = False) -> xr.DataArray:
        """refer to xarray dataarray method for docs"""
        return self._warehouse.where(condition, other=other, drop=drop)

    @overload
    def delete(
        self, ids: List[str], types: List[Type], inplace: bool = False
    ) -> xr.DataArray: ...

    @overload
    def delete(
        self, ids: List[str], types: List[Type], inplace: bool = True
    ) -> None: ...

    def delete(
        self, ids: List[str], types: List[Type], inplace: bool = False
    ) -> Optional[xr.DataArray]:
        if inplace:
            self._warehouse = self._warehouse.drop_sel(
                {self.id_coord: ids, self.type_coord: types}
            )
            return None
        else:
            return self._warehouse.drop_sel(
                {self.id_coord: ids, self.type_coord: types}
            )

    @overload
    def update(self, data: xr.DataArray, inplace: bool = True) -> None: ...

    @overload
    def update(self, data: xr.DataArray, inplace: bool = False) -> xr.DataArray: ...

    def update(
        self, data: xr.DataArray, inplace: bool = False
    ) -> Optional[xr.DataArray]:
        if inplace:
            self._warehouse.loc[data.coords] = data  # shape safe
            return None
        else:
            return self._warehouse.sel(data.coords)

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
    def _types(self) -> List[Type]:
        return self._warehouse.coords[self.type_coord].to_list()

    @property
    def _ids(self) -> List[str]:
        return self._warehouse.coords[self.id_coord].to_list()

    @property
    def id_to_type(self):
        return {id_: type_ for id_, type_ in zip(self._ids, self._types)}

    @property
    def type_to_id(self):
        dict_ = defaultdict(list)
        for type_, id_ in zip(self._types, self._ids):
            dict_[type_].append(id_)
        return dict_
