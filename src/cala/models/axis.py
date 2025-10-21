from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, Field

from cala.models.checks import has_no_nan, is_unique


class Axis:
    """Mixin providing common axis-related attributes."""

    frames_dim: str = "frame"
    height_dim: str = "height"
    width_dim: str = "width"
    component_dim: str = "component"
    """Name of the dimension representing individual components."""

    id_coord: str = "id_"
    timestamp_coord: str = "timestamp"
    detect_coord: str = "detected_on"
    frame_coord: str = "frame_idx"
    width_coord: str = "width"
    height_coord: str = "height"

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""
        return self.height_dim, self.width_dim

    @property
    def spatial_coords(self) -> tuple[str, str]:
        """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""
        return self.height_coord, self.width_coord

    @property
    def component_rename(self) -> dict[str, str]:
        return {
            AXIS.component_dim: f"{AXIS.component_dim}'",
            AXIS.id_coord: f"{AXIS.id_coord}'",
            AXIS.detect_coord: f"{AXIS.detect_coord}'",
        }


AXIS = Axis()


class Coord(BaseModel):
    name: str
    dtype: type
    dim: str | None = None
    checks: list[Callable] = Field(default_factory=list)


class Dim(BaseModel):
    name: str
    coords: list[Coord] = Field(default_factory=list)


class Coords(Enum):
    id = Coord(name=AXIS.id_coord, dtype=str, checks=[is_unique])
    height = Coord(name=AXIS.height_coord, dtype=int, checks=[is_unique])
    width = Coord(name=AXIS.width_coord, dtype=int, checks=[is_unique])
    frame = Coord(name=AXIS.frame_coord, dtype=int, checks=[is_unique])
    timestamp = Coord(name=AXIS.timestamp_coord, dtype=str, checks=[is_unique])
    detected = Coord(name=AXIS.detect_coord, dtype=int, checks=[has_no_nan])


class Dims(Enum):
    width = Dim(name=AXIS.width_dim, coords=[Coords.width.value])
    height = Dim(name=AXIS.height_dim, coords=[Coords.height.value])
    frame = Dim(name=AXIS.frames_dim, coords=[Coords.frame.value, Coords.timestamp.value])
    component = Dim(name=AXIS.component_dim, coords=[Coords.id.value, Coords.detected.value])
