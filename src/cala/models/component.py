from pydantic import BaseModel, Field
from cala.models.dim import Coord, Dim, Dims


class Component(BaseModel):
    name: str
    dims: tuple[Dim, ...]
    coords: list[Coord] = Field(default_factory=list)

    def model_post_init(self, __context__=None):
        for dim in self.dims:
            self.coords.extend(dim.coords)


footprint_schema = Component(name="footprint", dims=(Dims.width.value, Dims.height.value))
trace_schema = Component(name="trace", dims=(Dims.frame.value,))
