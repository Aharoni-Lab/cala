from pydantic import BaseModel, Field

from cala.models.dim import Coord, Dim, Dims
from cala.models.component import Component, footprint_schema, trace_schema


class Group(BaseModel):
    name: str
    component: Component
    dims: tuple[Dim, ...] = Field(default=tuple())
    coords: list[Coord] = Field(default_factory=list)

    def model_post_init(self, __context__=None) -> None:
        self.dims = self.component.dims + (Dims.component.value,)
        self.coords = self.component.coords + Dims.component.value.coords


footprint_group_schema = Group(
    name="footprint-group",
    component=footprint_schema,
)

trace_group_schema = Group(
    name="trace-group",
    component=trace_schema,
)
