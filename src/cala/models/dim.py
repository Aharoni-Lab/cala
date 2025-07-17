from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class Coord(BaseModel):
    name: str
    dtype: type
    dim: str | None = None


class Dim(BaseModel):
    name: str
    coords: list[Coord] = Field(default_factory=list)


class Coords(Enum):
    id = Coord(name="id", dtype=str)
    height = Coord(name="height", dtype=int)
    width = Coord(name="width", dtype=int)
    frame = Coord(name="frame", dtype=int)
    timestamp = Coord(name="timestamp", dtype=np.datetime64)
    confidence = Coord(name="confidence", dtype=float)


class Dims(Enum):
    width = Dim(name="width", coords=[Coords.width.value])
    height = Dim(name="height", coords=[Coords.height.value])
    frame = Dim(name="frame", coords=[Coords.frame.value, Coords.timestamp.value])
    component = Dim(name="component", coords=[Coords.id.value, Coords.confidence.value])
