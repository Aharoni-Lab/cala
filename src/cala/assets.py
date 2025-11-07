import contextlib
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator
from sparse import COO

from cala.config import config
from cala.models.axis import AXIS, Coords, Dims
from cala.models.checks import has_no_nan, is_non_negative
from cala.models.entity import Entity, Group
from cala.util import clear_dir

AssetType = TypeVar("AssetType", xr.DataArray, Path, None)


class Asset(BaseModel):
    validate_schema: bool = False
    array_: AssetType = None
    sparsify: ClassVar[bool] = False
    _entity: ClassVar[Entity]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @property
    def array(self) -> AssetType:
        return self.array_

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        if self.validate_schema:
            value.validate.against_schema(self._entity.model)
        if self.sparsify and isinstance(value.data, np.ndarray):
            value.data = COO.from_numpy(value.data)
        self.array_ = value

    @classmethod
    def from_array(cls, array: xr.DataArray) -> Self:
        if cls.sparsify and isinstance(array.data, np.ndarray):
            array.data = COO.from_numpy(array.data)
        return cls(array_=array)

    def reset(self) -> None:
        self.array_ = None

    def __eq__(self, other: "Asset") -> bool:
        return self.array.equals(other.array)

    @classmethod
    def entity(cls) -> Entity:
        return cls._entity

    @model_validator(mode="after")
    def validate_array_schema(self) -> Self:
        if self.validate_schema and self.array_ is not None:
            self.array_.validate.against_schema(self._entity.model)

        return self


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
            checks=[is_non_negative],
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
    sparsify = True


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
        peek_filter = {AXIS.frames_dim: slice(-self.peek_size, None)} if self.peek_size else None
        return self.full_array(isel_filter=peek_filter)

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        if self.zarr_path:
            if self.validate_schema:
                array.validate.against_schema(self._entity.model)
            array.to_zarr(self.zarr_path, mode="w")  # need to make sure it can overwrite
        else:
            self.array_ = array

    def reset(self) -> None:
        self.array_ = None
        if self.zarr_path:
            path = Path(self.zarr_path)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                contextlib.suppress(FileNotFoundError)

    def full_array(self, isel_filter: dict = None, sel_filter: dict = None) -> xr.DataArray:
        if self.zarr_path:
            try:
                return self.load_zarr(isel_filter=isel_filter, sel_filter=sel_filter).compute()
            except FileNotFoundError:
                pass
        return (
            self.array_.isel(isel_filter).sel(sel_filter)
            if self.array_ is not None
            else self.array_
        )

    def load_zarr(self, isel_filter: dict = None, sel_filter: dict = None) -> xr.DataArray:
        da = (
            xr.open_zarr(self.zarr_path)
            .isel(isel_filter)
            .sel(sel_filter)
            .to_dataarray()
            .drop_vars(["variable"])
            .isel(variable=0)
        )
        return da.assign_coords(
            {
                AXIS.id_coord: lambda ds: da[AXIS.id_coord].astype(str),
                AXIS.timestamp_coord: lambda ds: da[AXIS.timestamp_coord].astype(str),
            }
        )

    def update(self, array: xr.DataArray, **kwargs: Any) -> None:
        if self.validate_schema:
            array.validate.against_schema(self._entity.model)
        array.to_zarr(self.zarr_path, **kwargs)

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
            checks=[is_non_negative],
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
    sparsify = True


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


class Buffer(Asset):
    """
    Implements a fake ring buffer to avoid expensive copying that occurs with
    numpy concat, append, and stack.

    Works by preallocating a space twice the desired size.
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

    validate_schema: bool = False
    """Validation currently does not play nicely with this class."""

    size: int
    _full: bool = PrivateAttr(False)
    _next: int = PrivateAttr(default=0)

    def append(self, array: xr.DataArray) -> None:
        self.array_.data[self._next] = array.data
        self.array_.data[self._next + self.size] = array.data
        for coord in [AXIS.frame_coord, AXIS.timestamp_coord]:
            self.array_[coord].data[self._next] = array[coord].item()
            self.array_[coord].data[self._next + self.size] = array[coord].item()

        self._next = (self._next + 1) % self.size
        if not self._full:
            # check if this made the buffer full
            self._full = self._next == 0

    @property
    def array(self) -> xr.DataArray | None:
        if self.array_ is None:
            return None
        if self._full:
            out = self.array_.isel({AXIS.frames_dim: slice(self._next, self._next + self.size)})
        else:
            out = self.array_.isel({AXIS.frames_dim: slice(None, self._next)})
        # kinda expensive. maybe float is fine?
        return out  # .assign_coords({AXIS.frame_coord: out[AXIS.frame_coord].astype(int)})

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        """
        Build a new buffer array.
        """
        array = (
            array.volumize.dim_with_coords(
                dim=AXIS.frames_dim, coords=[AXIS.frame_coord, AXIS.timestamp_coord]
            )
            if AXIS.frames_dim not in array.dims
            else array.isel({AXIS.frames_dim: slice(-self.size, None)})
        )
        fill_sizes = dict(array.sizes)
        fill_sizes[AXIS.frames_dim] = self.size - array.sizes[AXIS.frames_dim]
        fill = np.zeros(list(fill_sizes.values()))
        filler = xr.DataArray(
            fill,
            dims=array.dims,
            coords={
                AXIS.frame_coord: (AXIS.frames_dim, [np.nan] * (fill_sizes[AXIS.frames_dim])),
                AXIS.timestamp_coord: (AXIS.frames_dim, [""] * (fill_sizes[AXIS.frames_dim])),
            },
        )
        buffer = xr.concat([array, filler] * 2, dim=AXIS.frames_dim)

        self._full = array.sizes[AXIS.frames_dim] >= self.size
        self._next = np.min((array.sizes[AXIS.frames_dim], self.size)) % self.size
        self.array_ = buffer

    @classmethod
    def from_array(cls, array: xr.DataArray, size: int) -> Self:
        buffer = cls(size=size)
        buffer.array = array
        return buffer
