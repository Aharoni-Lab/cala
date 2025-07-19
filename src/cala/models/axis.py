from enum import Enum

from pydantic import BaseModel, Field


class Axis:
    """Mixin providing common axis-related attributes."""

    frames_dim: str = "frame"
    height_dim: str = "height"
    width_dim: str = "width"
    component_dim: str = "component"
    """Name of the dimension representing individual components."""

    id_coord: str = "id_"
    type_coord: str = "type_"
    timestamp_coord: str = "timestamp"
    confidence_coord: str = "confidence"
    frame_coord: str = "frame"
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


AXIS = Axis()


class Coord(BaseModel):
    name: str
    dtype: type
    dim: str | None = None


class Dim(BaseModel):
    name: str
    coords: list[Coord] = Field(default_factory=list)


class Coords(Enum):
    id = Coord(name=AXIS.id_coord, dtype=str)
    height = Coord(name=AXIS.height_coord, dtype=int)
    width = Coord(name=AXIS.width_coord, dtype=int)
    frame = Coord(name=AXIS.frame_coord, dtype=int)
    timestamp = Coord(name=AXIS.timestamp_coord, dtype=str)
    confidence = Coord(name=AXIS.confidence_coord, dtype=float)


class Dims(Enum):
    width = Dim(name=AXIS.width_dim, coords=[Coords.width.value])
    height = Dim(name=AXIS.height_dim, coords=[Coords.height.value])
    frame = Dim(name=AXIS.frames_dim, coords=[Coords.frame.value, Coords.timestamp.value])
    component = Dim(name=AXIS.component_dim, coords=[Coords.id.value, Coords.confidence.value])
