from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from xarray_validate import CoordsSchema, DataArraySchema, DimsSchema, DTypeSchema

from cala.models.axis import Coord, Dim, Dims


class Component(Enum):
    """Enumeration of possible component types in the imaging data.

    Attributes:
        NEURON: Represents neuronal components.
        BACKGROUND: Represents background components (non-neuronal signals).
    """

    NEURON = "neuron"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


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
            for coord in dim.coords:
                coord.dim = dim.name
            self.coords.extend(dim.coords)

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
        return CoordsSchema(
            {
                c.name: DataArraySchema(
                    dims=DimsSchema((c.dim,)), dtype=DTypeSchema(c.dtype), checks=c.checks
                )
                for c in coords
            }
        )


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
