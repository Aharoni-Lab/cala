from typing import ClassVar

import xarray as xr
from pydantic import BaseModel, PrivateAttr, field_validator
from xarray_validate import CoordsSchema, DataArraySchema, DimsSchema, DTypeSchema

from cala.models.axis import Coord, Dims
from cala.models.entity import Entity, Group


class Observable(BaseModel):
    array: xr.DataArray
    _model: ClassVar[Entity]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def model(cls) -> Entity:
        return cls._model

    @field_validator("array", mode="after")
    @classmethod
    def validate_array_schema(cls, value: xr.DataArray) -> None:
        schema = cls._build_entity_schema(cls._model)
        value.validate.against_schema(schema)

    @staticmethod
    def _build_coord_schema(coords: list[Coord]) -> CoordsSchema:
        return CoordsSchema(
            {
                c.name: DataArraySchema(dims=DimsSchema((c.dim,)), dtype=DTypeSchema(c.dtype))
                for c in coords
            }
        )

    @classmethod
    def _build_entity_schema(cls, schema: Entity) -> DataArraySchema:
        coords_schema = cls._build_coord_schema(schema.coords) if schema.coords else None

        return DataArraySchema(
            dims=DimsSchema(tuple(dim.name for dim in schema.dims), ordered=False),
            coords=coords_schema,
            dtype=DTypeSchema(schema.dtype),
        )


class Footprint(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(
        Entity(name="footprint", dims=(Dims.width.value, Dims.height.value), dtype=float)
    )


class Trace(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(
        Entity(name="trace", dims=(Dims.frame.value,), dtype=float)
    )


class Frame(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="frame", dims=(Dims.width.value, Dims.height.value, Dims.frame.value), dtype=float
        )
    )


class Footprints(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(
        Group(name="footprint-group", entity=Footprint.model(), group_by=Dims.component)
    )


class Traces(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(
        Group(name="trace-group", entity=Trace.model(), group_by=Dims.component)
    )


class Movie(Observable):
    array: xr.DataArray
    _model: ClassVar[Entity] = PrivateAttr(Group(name="movie", entity=Frame.model()))
