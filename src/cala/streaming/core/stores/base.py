from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Set, Hashable, overload, Dict

import numpy as np
import xarray as xr


@dataclass
class BaseStore(ABC):
    dimensions: Tuple[str, ...]
    component_dimension: str

    id_coordinate: str = "id_"
    type_coordinate: str = "type_"

    _warehouse: xr.DataArray = field(default_factory=lambda: xr.DataArray())

    def __post_init__(self):
        if self.component_dimension not in self.dimensions:
            raise ValueError(
                f"component_axis {self.component_dimension} must be in dims {self.dimensions}."
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
        data = xr.DataArray(
            data_array,
            dims=self.dimensions,
            coords={
                self.type_coordinate: ([self.component_dimension], types),
                self.id_coordinate: ([self.component_dimension], ids),
            },
        )
        return data.set_index(
            {self.component_dimension: [self.type_coordinate, self.id_coordinate]}
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
                f"Inappropriate storage dimensions. Values in this storage must have:\n\tdimensions: {self.dimensions}\n\tcoordinates:{self.id_coordinate, self.type_coordinate} that are attached to {self.component_dimension}.\nRefer to generate_warehouse method for formatting an unlabeled array."
            )

    def _validate_dims(self, dimensions: Tuple[Hashable, ...]) -> bool:
        return set(self.dimensions) == set(dimensions)

    def _validate_coords(self, coords: Set) -> bool:
        """this also validates the coordinate attachment to index.

        Args:
            coords:

        Returns:

        """
        return {
            self.id_coordinate,
            self.type_coordinate,
            self.component_dimension,
        } == coords

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
                [self._warehouse, to_insert], dim=self.component_dimension
            )
        else:
            return xr.concat([self._warehouse, to_insert], dim=self.component_dimension)

    def slice(
        self,
        ids: List[str],
        types: List[str],
    ) -> xr.DataArray:
        # args = {self.id_coordinate: ids, self.type_coordinate: types}
        # coords = {coord: idx for coord, idx in args.items() if idx is not None}

        # return self._warehouse.sel(coords)
        results = []
        if not ids:
            for group, members in self.type_to_id.items():
                if group in types:
                    results.append(
                        self._warehouse.sel({self.type_coordinate: group}).sel(
                            {self.id_coordinate: members}
                        )
                    )
        else:
            input_types = [self.id_to_type[id_] for id_ in ids]
            groups = defaultdict(list)
            for id_, type_ in zip(ids, input_types):
                groups[type_].append(id_)
            for group, members in groups.items():
                if not types:
                    results.append(
                        self._warehouse.sel({self.type_coordinate: group}).sel(
                            {self.id_coordinate: members}
                        )
                    )
                else:
                    if group in types:
                        results.append(
                            self._warehouse.sel({self.type_coordinate: group}).sel(
                                {self.id_coordinate: members}
                            )
                        )

        if not results:
            raise KeyError("No elements match the search criteria.")
        if len(results) == 1:
            return results[0]
        else:
            return xr.concat(results, dim=self.component_dimension)

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
        self, ids: List[str], types: List[str], inplace: bool = False
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
        keep_ids = set(self._ids) - set(ids)
        keep_types = set(self._types) - set(types)
        # we delete intersection, not union
        for ids in [self.type_to_id[type_] for type_ in keep_types]:
            keep_ids.update(ids)
        keep_ids = list(keep_ids)

        if inplace:
            self._warehouse = self._warehouse.sel({self.id_coordinate: keep_ids})
        else:
            return self._warehouse.sel({self.id_coordinate: keep_ids})

    @overload
    def update(self, data: xr.DataArray, inplace: bool = True) -> None: ...

    @overload
    def update(self, data: xr.DataArray, inplace: bool = False) -> xr.DataArray: ...

    def update(
        self, data: xr.DataArray, inplace: bool = False
    ) -> Optional[xr.DataArray]:
        """only allows formatted dataarray with appropriate dims and coords. can use the generate_warehouse method beforehand."""
        id_ = self.id_coordinate

        data_coords = data.coords[self.id_coordinate].values.tolist()

        id_indices = (
            self.warehouse[self.id_coordinate]
            .values.tolist()
            .index(
                self.warehouse.sel({self.id_coordinate: data_coords[1]})[
                    self.id_coordinate
                ]
            )
        )
        if inplace:
            self._warehouse.loc[data_coords] = data  # shape safe
            return None
        else:
            return self._warehouse.sel(data_coords)

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
        return self._warehouse.coords[self.type_coordinate].values.tolist()

    @property
    def _ids(self) -> List[str]:
        return self._warehouse.coords[self.id_coordinate].values.tolist()

    @property
    def id_to_type(self) -> Dict[str, str]:
        return {id_: type_ for id_, type_ in zip(self._ids, self._types)}

    @property
    def type_to_id(self) -> Dict[str, List[str]]:
        dict_ = defaultdict(list)
        for type_, id_ in zip(self._types, self._ids):
            dict_[type_].append(id_)
        return dict_
