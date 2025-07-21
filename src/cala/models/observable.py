from typing import ClassVar

import xarray as xr
from pydantic import BaseModel, PrivateAttr, field_validator

from cala.models.axis import Dims
from cala.models.checks import is_non_negative
from cala.models.entity import Entity, Group


class Observable(BaseModel):
    array: xr.DataArray
    _entity: ClassVar[Entity]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def entity(cls) -> Entity:
        return cls._entity

    @field_validator("array", mode="after")
    @classmethod
    def validate_array_schema(cls, value: xr.DataArray) -> xr.DataArray:
        value.validate.against_schema(cls._entity.model)

        return value


class Footprint(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="footprint",
            dims=(Dims.width.value, Dims.height.value),
            dtype=float,
            checks=[is_non_negative],
        )
    )


class Trace(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(name="trace", dims=(Dims.frame.value,), dtype=float, checks=[is_non_negative])
    )


class Frame(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="frame",
            dims=(Dims.width.value, Dims.height.value, Dims.frame.value),
            dtype=float,
            checks=[is_non_negative],
        )
    )


class Footprints(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="footprint-group",
            entity=Footprint.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Traces(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="trace-group",
            entity=Trace.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Movie(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="movie",
            entity=Frame.entity(),
            checks=[is_non_negative],
        )
    )
