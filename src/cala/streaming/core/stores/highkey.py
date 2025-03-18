from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set, Hashable, overload, Dict

import numpy as np
import xarray as xr

from .midkey import MidkeyStore


@dataclass(kw_only=True)
class HighkeyStore(MidkeyStore):
    """These stores care about the types of components, and has functionalities to
    ingest / output parts of warehouse based on component type."""

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
        self._warehouse = self.generate_warehouse(data_array=empty_array)

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

    @property
    def _types(self) -> List[str]:
        return self._warehouse.coords[self.type_coord].values.tolist()

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
