from pydantic import Field

from cala.models.dim import Dim, Dims
from cala.models.entity import Entity, footprint_schema, trace_schema, frame_schema


class Group(Entity):
    """
    an xarray dataarray entity that is also a group of entities.
    """

    entity: Entity
    group_by: Dims | None = None
    dims: tuple[Dim, ...] = Field(default=tuple())

    def model_post_init(self, __context__=None) -> None:
        if self.group_by:
            self.dims = self.entity.dims + (self.group_by.value,)
            self.coords = self.entity.coords + self.group_by.value.coords


footprint_group_schema = Group(
    name="footprint-group", entity=footprint_schema, group_by=Dims.component
)

trace_group_schema = Group(name="trace-group", entity=trace_schema, group_by=Dims.component)

movie_schema = Group(name="movie", entity=frame_schema)
