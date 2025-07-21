from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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

    def model_post_init(self, __context__: None = None) -> None:
        for dim in self.dims:
            for coord in dim.coords:
                coord.dim = dim.name
            self.coords.extend(dim.coords)


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
