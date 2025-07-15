from pydantic import BaseModel, Field
from cala.models.dim import Coord, Dim, Dims


class Entity(BaseModel):
    """
    A base entity describable with an xarray dataarray.
    """

    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)
    dtype: type

    def model_post_init(self, __context__=None):
        for dim in self.dims:
            for coord in dim.coords:
                coord.dim = dim.name
            self.coords.extend(dim.coords)


footprint_schema = Entity(name="footprint", dims=(Dims.width.value, Dims.height.value), dtype=float)
trace_schema = Entity(name="trace", dims=(Dims.frame.value,), dtype=float)
frame_schema = Entity(
    name="frame", dims=(Dims.width.value, Dims.height.value, Dims.frame.value), dtype=float
)
