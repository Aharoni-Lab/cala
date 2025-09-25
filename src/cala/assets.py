import contextlib
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import xarray as xr
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator

from cala.config import config
from cala.models.axis import AXIS, Coords, Dims
from cala.models.checks import has_no_nan, is_non_negative
from cala.models.entity import Entity, Group
from cala.util import clear_dir

AssetType = TypeVar("AssetType", xr.DataArray, Path, None)


class Asset(BaseModel):
    array_: AssetType = None
    _entity: ClassVar[Entity]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @property
    def array(self) -> AssetType:
        return self.array_

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        self.array_ = value

    @classmethod
    def from_array(cls, array: xr.DataArray) -> "Asset":
        return cls(array_=array)

    def reset(self) -> None:
        self.array_ = None

    def __eq__(self, other: "Asset") -> bool:
        return self.array.equals(other.array)

    @classmethod
    def entity(cls) -> Entity:
        return cls._entity

    @field_validator("array_", mode="after")
    @classmethod
    def validate_array_schema(cls, value: xr.DataArray) -> AssetType:
        if value is not None:
            value.validate.against_schema(cls._entity.model)

        return value


class Footprint(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="footprint",
            dims=(Dims.width.value, Dims.height.value),
            dtype=float,
            checks=[is_non_negative, has_no_nan],
        )
    )


class Trace(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="trace",
            dims=(Dims.frame.value,),
            dtype=float,
            checks=[is_non_negative, has_no_nan],
        )
    )


class Frame(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="frame",
            dims=(Dims.width.value, Dims.height.value),
            dtype=None,  # np.number,  # gets converted to float64 in xarray-validate
            checks=[is_non_negative, has_no_nan],
        )
    )


class Footprints(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="footprint-group",
            member=Footprint.entity(),
            group_by=Dims.component,
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
        )
    )


class Traces(Asset):
    zarr_path: Path | None = None
    """relative to config.user_data_dir"""
    peek_size: int | None = None
    """
    Traces(array=array, path=path) -> saves to zarr (should be set in this asset, and leave
    untouched in nodes.)
    Traces.array -> loads from zarr
    """

    @property
    def array(self) -> xr.DataArray:
        if self.zarr_path:
            try:
                da = (
                    xr.open_zarr(self.zarr_path)
                    .isel({AXIS.frames_dim: slice(-self.peek_size, None)})
                    .to_dataarray()
                    .drop_vars(["variable"])
                    .isel(variable=0)
                )
                return da.assign_coords(
                    {
                        AXIS.id_coord: lambda ds: da[AXIS.id_coord].astype(str),
                        AXIS.timestamp_coord: lambda ds: da[AXIS.timestamp_coord].astype(str),
                    }
                ).compute()

            except FileNotFoundError:
                return self.array_
        else:
            return self.array_

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        if self.zarr_path:
            array.to_zarr(self.zarr_path, mode="w")  # need to make sure it can overwrite
        else:
            self.array_ = array

    def update(self, array: xr.DataArray, **kwargs: Any) -> None:
        self.validate_array_schema(array)
        array.to_zarr(self.zarr_path, **kwargs)

    def reset(self) -> None:
        self.array_ = None
        if self.zarr_path:
            path = Path(self.zarr_path)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                contextlib.suppress(FileNotFoundError)

    @classmethod
    def from_array(
        cls, array: xr.DataArray, zarr_path: Path | str | None = None, peek_size: int | None = None
    ) -> "Traces":
        new_cls = cls(zarr_path=zarr_path, peek_size=peek_size)
        new_cls.array = array
        return new_cls

    @field_validator("zarr_path", mode="after")
    @classmethod
    def validate_zarr_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return value
        zarr_dir = (config.user_dir / value).resolve()
        zarr_dir.mkdir(parents=True, exist_ok=True)
        clear_dir(zarr_dir)
        return zarr_dir

    @model_validator(mode="after")
    def check_zarr_setting(self) -> "Traces":
        if self.zarr_path:
            assert self.peek_size, "peek_size must be set for zarr."
        return self

    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="trace-group",
            member=Trace.entity(),
            group_by=Dims.component,
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
        )
    )


class Movie(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="movie",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
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
            checks=[is_non_negative, has_no_nan],
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
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
        )
    )


class PixStats(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="pix-stat",
            dims=(Dims.width.value, Dims.height.value, Dims.component.value),
            dtype=float,
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
        )
    )


class Overlaps(Asset):
    _entity: ClassVar[Entity] = PrivateAttr(
        Entity(
            name="overlap",
            dims=comp_dims,
            dtype=bool,
            checks=[has_no_nan],
            allow_extra_coords=False,
        )
    )


class Residual(Asset):
    """
    Computes and maintains a buffer of residual signals.

    This method implements the residual computation by subtracting the
    reconstructed signal from the original data. It maintains only the
    most recent frames as specified by the buffer length.

    The residual buffer contains the recent history of unexplained variance
    in the data after accounting for known components.
    """

    _entity: ClassVar[Entity] = PrivateAttr(
        Group(
            name="frame",
            member=Frame.entity(),
            group_by=Dims.frame.value,
            checks=[is_non_negative, has_no_nan],
            allow_extra_coords=False,
        )
    )
