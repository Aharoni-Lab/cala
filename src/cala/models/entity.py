from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from cala.models.axis import Coord, Dim, Dims


class Entity(BaseModel):
    """
    A base entity describable with an xarray dataarray.
    """

    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)
    dtype: type

    def model_post_init(self, __context__: None = None) -> None:
        for dim in self.dims:
            for coord in dim.coords:
                coord.dim = dim.name
            self.coords.extend(dim.coords)


class Entities(Enum):
    footprint = Entity(name="footprint", dims=(Dims.width.value, Dims.height.value), dtype=float)
    trace = Entity(name="trace", dims=(Dims.frame.value,), dtype=float)
    frame = Entity(
        name="frame", dims=(Dims.width.value, Dims.height.value, Dims.frame.value), dtype=float
    )


class Group(Entity):
    """
    an xarray dataarray entity that is also a group of entities.
    """

    entity: Entity
    group_by: Dims | None = None
    dims: tuple[Dim, ...] = Field(default=tuple())
    dtype: type = Field(default=Any)

    def model_post_init(self, __context__: None = None) -> None:
        if self.group_by:
            self.dims = self.entity.dims + (self.group_by.value,)
            self.coords = self.entity.coords + self.group_by.value.coords
        self.dtype = self.entity.dtype


class Groups(Enum):
    footprint = Group(
        name="footprint-group", entity=Entities.footprint.value, group_by=Dims.component
    )
    trace = Group(name="trace-group", entity=Entities.trace.value, group_by=Dims.component)
    movie = Group(name="movie", entity=Entities.frame.value)
