from copy import deepcopy
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
            dims=(Dims.width.value, Dims.height.value),
            dtype=float,
            checks=[is_non_negative],
        )
    )


class Footprints(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="footprint-group",
            member=Footprint.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Traces(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="trace-group",
            member=Trace.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Movie(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="movie",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative],
        )
    )


comp_dims = (Dims.component.value, deepcopy(Dims.component.value))
comp_dims[1].name += "'"
for coord in comp_dims[1].coords:
    coord.name += "'"


class CompStat(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="comp-stat",
            dims=comp_dims,
            dtype=float,
            checks=[],
        )
    )


class PixStat(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="pix-stat",
            dims=(Dims.width.value, Dims.height.value, Dims.component.value),
            dtype=float,
            checks=[],
        )
    )


class Overlap(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="overlap",
            dims=comp_dims,
            dtype=int,
            checks=[],
        )
    )


class Residual(Observable):
    array: xr.DataArray
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="frame",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative],
        )
    )
