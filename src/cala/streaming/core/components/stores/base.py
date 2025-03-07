from collections import defaultdict
from dataclasses import dataclass
from typing import List, Type, Optional

import numpy as np
import xarray as xr


@dataclass
class BaseStore:
    _dataset: xr.Dataset = None
    dataset_name: str = "dataset"
    component_axis: str = "components"
    type_coord: str = "type_"
    id_coord: str = "id_"
    width_axis: str = "width"
    height_axis: str = "height"

    def __post_init__(self):
        empty_array = np.empty(shape=(0, 0, 0))
        empty_ids = []
        empty_types = []
        self.set_array(data_array=empty_array, ids=empty_ids, types=empty_types)

    @property
    def array(self) -> xr.DataArray:
        return self._dataset.to_dataarray()

    def _generate_dataset(
        self, data_array: np.ndarray | xr.DataArray, ids: List[str], types: List[Type]
    ) -> xr.Dataset:
        # validate if sizes match
        return xr.Dataset(
            {
                self.dataset_name: (
                    [self.width_axis, self.height_axis, self.component_axis],
                    data_array,
                )
            },
            coords={
                self.type_coord: ([self.component_axis], types),
                self.id_coord: ([self.component_axis], ids),
            },
        )

    def set_array(
        self, data_array: np.ndarray | xr.DataArray, ids: List[str], types: List[Type]
    ) -> None:
        self._dataset = self._generate_dataset(data_array, ids, types)

    def slice_by_coordinates(
        self,
        ids: Optional[List[str]],
        types: Optional[List[Type]],
        width: Optional[List[int]],
        height: Optional[List[int]],
    ) -> xr.Dataset:
        return self._dataset.sel(
            {
                self.id_coord: ids,
                self.type_coord: types,
                self.width_axis: width,
                self.height_axis: height,
            }
        )

    def insert_array(
        self, data_array: np.ndarray | xr.DataArray, ids: List[str], types: List[Type]
    ):
        to_insert = self._generate_dataset(data_array, ids, types)
        self._dataset = xr.concat([self._dataset, to_insert], dim=self.component_axis)

    @property
    def _types(self) -> List[Type]:
        return self._dataset.coords[self.type_coord].to_list()

    @property
    def _ids(self) -> List[str]:
        return self._dataset.coords[self.id_coord].to_list()

    @property
    def id_to_type(self):
        return {id_: type_ for id_, type_ in zip(self._ids, self._types)}

    @property
    def type_to_id(self):
        dict_ = defaultdict(list)
        for type_, id_ in zip(self._types, self._ids):
            dict_[type_].append(id_)
        return dict_

    def ids_to_types(self, ids: List[str]) -> List[Type]:
        if not set(self._ids).issuperset(set(ids)):
            raise KeyError("Trying to access IDs that are not present in the array.")
        return [type_ for id_, type_ in self.id_to_type.items() if id_ in ids]
