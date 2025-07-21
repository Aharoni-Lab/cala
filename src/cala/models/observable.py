from typing import Annotated, get_args

import xarray as xr
from pydantic import BaseModel, field_validator

from cala.models.axis import Dims
from cala.models.entity import Entity, Group


class Observable(BaseModel):
    array: Annotated[xr.DataArray, Entity]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @field_validator("array", mode="after")
    @classmethod
    def validate_array_schema(cls, value: xr.DataArray) -> None:
        value.validate.against_schema(get_args(cls.array)[1])


TFootprint: type = Annotated[
    xr.DataArray, Entity(name="footprint", dims=(Dims.width.value, Dims.height.value), dtype=float)
]

TTrace: type = Annotated[xr.DataArray, Entity(name="trace", dims=(Dims.frame.value,), dtype=float)]

TFrame: type = Annotated[
    xr.DataArray,
    Entity(name="frame", dims=(Dims.width.value, Dims.height.value, Dims.frame.value), dtype=float),
]

TFootprints: type = Annotated[
    xr.DataArray,
    Group(name="footprint-group", entity=get_args(TFootprint)[1], group_by=Dims.component),
]

TTraces: type = Annotated[
    xr.DataArray, Group(name="trace-group", entity=get_args(TTrace)[1], group_by=Dims.component)
]

TMovie: type = Annotated[xr.DataArray, Group(name="movie", entity=get_args(TFrame)[1])]


class Footprint(Observable):
    array: TFootprint


class Trace(Observable):
    array: TTrace


class Frame(Observable):
    array: TFrame


class Footprints(Observable):
    array: TFootprints


class Traces(Observable):
    array: TTraces


class Movie(Observable):
    array: TMovie
