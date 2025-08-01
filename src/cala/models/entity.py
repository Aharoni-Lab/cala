from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from xarray_validate import CoordsSchema, DataArraySchema, DimsSchema, DTypeSchema

from cala.models.axis import Coord, Dim, Dims


class Entity(BaseModel):
    """
    A base entity describable with an xarray dataarray.
    """

    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)
    dtype: type
    checks: list[Callable] = Field(default_factory=list)

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
            dtype=DTypeSchema(self.dtype),
            checks=self.checks,
        )

    @staticmethod
    def _build_coord_schema(coords: list[Coord]) -> CoordsSchema:
        spec = dict()

        for c in coords:
            dim = DimsSchema((c.dim,)) if c.dim else None
            spec[c.name] = DataArraySchema(dims=dim, dtype=DTypeSchema(c.dtype), checks=c.checks)

        return CoordsSchema(spec)


class Group(Entity):
    """
    an xarray dataarray entity that is also a group of entities.
    """

    member: Entity
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
