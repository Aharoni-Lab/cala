from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import xarray as xr
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from cala.models.axis import AXIS, Coords, Dims
from cala.models.checks import is_non_negative
from cala.models.entity import Entity, Group


class Asset(BaseModel):
    array_: xr.DataArray | None = None
    _entity: ClassVar[Entity]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @property
    def array(self) -> xr.DataArray | None:
        return self.array_

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        self.array_ = value

    @classmethod
    def from_array(cls, array: xr.DataArray) -> "Asset":
        return cls(array_=array)

    def __eq__(self, other: "Asset") -> bool:
        return self.array.equals(other.array)

    @classmethod
    def entity(cls) -> Entity:
        return cls._entity

    @field_validator("array_", mode="after")
    @classmethod
    def validate_array_schema(cls, value: xr.DataArray) -> xr.DataArray:
        value.validate.against_schema(cls._entity.model)

        return value


class Footprint(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="footprint",
            dims=(Dims.width.value, Dims.height.value),
            dtype=float,
            checks=[is_non_negative],
        )
    )


class Trace(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(name="trace", dims=(Dims.frame.value,), dtype=float, checks=[is_non_negative])
    )


class Frame(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="frame",
            dims=(Dims.width.value, Dims.height.value),
            dtype=float,
            checks=[is_non_negative],
        )
    )


class Footprints(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="footprint-group",
            member=Footprint.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Traces(Asset):
    zarr_path: Path | str | None = None
    peek_size: int | None = None
    """
    Traces(array=array, path=path) -> saves to zarr (should be set in this asset, and leave
    untouched in nodes.)
    Traces.array -> loads from zarr
    """

    @property
    def array(self) -> xr.DataArray:
        if self.zarr_path:
            return (
                xr.open_zarr(self.zarr_path)
                .isel({AXIS.frames_dim: slice(-self.peek_size, None)})
                .to_dataarray()
                .isel({"variable": 0})  # not sure why it automatically makes this coordinate
                .reset_coords("variable", drop=True)
            )
        else:
            return self.array_

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        if self.zarr_path:
            array.to_zarr(self.zarr_path, mode="w")  # need to make sure it can overwrite
        else:
            self.array_ = array

    def append(self, array: xr.DataArray, dim: str | list[str]) -> None:
        array.to_zarr(self.zarr_path, append_dim=dim)

    @classmethod
    def from_array(
        cls, array: xr.DataArray, zarr_path: Path | str | None = None, peek_size: int | None = None
    ) -> "Traces":
        return cls(array_=array, zarr_path=zarr_path, peek_size=peek_size)

    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="trace-group",
            member=Trace.entity(),
            group_by=Dims.component,
            checks=[is_non_negative],
        )
    )


class Movie(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="movie",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative],
        )
    )


class PopSnap(Asset):
    """
    A snapshot of a population trait.

    Mainly used for Traces that only has one frame.
    """

    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="pop-snap",
            dims=(Dims.component.value,),
            dtype=float,
            coords=[Coords.frame.value, Coords.timestamp.value],
        )
    )


comp_dims = (Dims.component.value, deepcopy(Dims.component.value))
comp_dims[1].name += "'"
for coord in comp_dims[1].coords:
    coord.name += "'"


class CompStats(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="comp-stat",
            dims=comp_dims,
            dtype=float,
            checks=[],
        )
    )


class PixStats(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="pix-stat",
            dims=(Dims.width.value, Dims.height.value, Dims.component.value),
            dtype=float,
            checks=[],
        )
    )


class Overlaps(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="overlap",
            dims=comp_dims,
            dtype=bool,
            checks=[],
        )
    )


class Residual(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="frame",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative],
        )
    )
