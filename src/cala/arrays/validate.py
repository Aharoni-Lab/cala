from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field, PrivateAttr
from xarray_validate import CoordsSchema, DataArraySchema, DimsSchema, DTypeSchema

from cala.arrays import AXIS


def is_non_negative(da: xr.DataArray) -> None:
    if da.min() < 0:
        raise ValueError("Array is not non-negative")


def is_unique(da: xr.DataArray) -> None:
    elem, counts = np.unique(da, return_counts=True)
    if counts.max() > 1:
        raise ValueError(f"The values in DataArray are not unique : {elem[counts > 1]}")


def is_unit_interval(da: xr.DataArray) -> None:
    if da.min() < 0 or da.max() > 1:
        raise ValueError("The values in DataArray are not unit interval.")


def has_no_nan(da: xr.DataArray) -> None:
    if np.isnan(da).any():
        raise ValueError("The DataArray has nan values.")


class Coord(BaseModel):
    name: str
    dtype: type
    dim: str | None = None
    checks: list[Callable] = Field(default_factory=list)


class Dim(BaseModel):
    name: str
    coords: list[Coord] = Field(default_factory=list)


class Coords(Enum):
    id = Coord(name=AXIS.id_coord, dtype=str, checks=[is_unique])
    height = Coord(name=AXIS.height_coord, dtype=int, checks=[is_unique])
    width = Coord(name=AXIS.width_coord, dtype=int, checks=[is_unique])
    frame = Coord(name=AXIS.frame_coord, dtype=int, checks=[is_unique])
    timestamp = Coord(name=AXIS.timestamp_coord, dtype=str, checks=[is_unique])
    detected = Coord(name=AXIS.detect_coord, dtype=int, checks=[has_no_nan])


class Dims(Enum):
    width = Dim(name=AXIS.width_dim, coords=[Coords.width.value])
    height = Dim(name=AXIS.height_dim, coords=[Coords.height.value])
    frame = Dim(name=AXIS.frame_dim, coords=[Coords.frame.value, Coords.timestamp.value])
    component = Dim(name=AXIS.component_dim, coords=[Coords.id.value, Coords.detected.value])


class Schema(BaseModel):
    """
    Wrapper around xarray-schema

    """

    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)
    dtype: type | None
    checks: list[Callable] = Field(default_factory=list)
    allow_extra_coords: bool = True

    _model: DataArraySchema = PrivateAttr(DataArraySchema())

    @property
    def model(self) -> DataArraySchema:
        return self._model

    def model_post_init(self, __context__: None = None) -> None:
        for dim in self.dims:
            coords = []
            for coord in dim.coords:
                c = deepcopy(coord)
                c.dim = dim.name
                coords.append(c)
            dim.coords = coords
            self.coords.extend(coords)

        self._model = self.to_schema()

    def to_schema(self) -> DataArraySchema:
        coords_schema = self._build_coord_schema(self.coords) if self.coords else None

        return DataArraySchema(
            dims=DimsSchema(tuple(dim.name for dim in self.dims), ordered=False),
            coords=coords_schema,
            dtype=DTypeSchema(self.dtype) if self.dtype else None,
            checks=self.checks,
        )

    def _build_coord_schema(self, coords: list[Coord]) -> CoordsSchema:
        spec = dict()

        for c in coords:
            dim = DimsSchema((c.dim,)) if c.dim else None
            spec[c.name] = DataArraySchema(dims=dim, dtype=DTypeSchema(c.dtype), checks=c.checks)

        return CoordsSchema(spec, allow_extra_keys=self.allow_extra_coords)


class Bundle(Schema):
    """
    an xarray dataarray entity that is also a group of entities.
    """

    member: Schema
    group_by: Dims | None = None
    dims: tuple[Dim, ...] = Field(default=tuple())
    dtype: type = Field(default=Any)

    def model_post_init(self, __context__: None = None) -> None:
        self.dims = self.member.dims
        self.coords = self.member.coords

        if self.group_by:
            self.dims += (self.group_by.value,)
            self.coords += self.group_by.value.coords

        self.dtype = self.member.dtype

        self._model = self.to_schema()
