from pydantic import BaseModel, Field
from cala.models.dim import Coord, Dim, Dims


class Entity(BaseModel):
    """
    A base entity describable with an xarray dataarray.
    """

    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)

    def model_post_init(self, __context__=None):
        for dim in self.dims:
            self.coords.extend(dim.coords)


footprint_schema = Entity(name="footprint", dims=(Dims.width.value, Dims.height.value))
trace_schema = Entity(name="trace", dims=(Dims.frame.value,))
frame_schema = Entity(name="frame", dims=(Dims.width.value, Dims.height.value, Dims.frame.value))
