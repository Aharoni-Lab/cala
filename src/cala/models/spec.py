from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from cala.config.yaml import ConfigYAMLMixin


class NodeSpec(BaseModel):
    id: str
    params: dict[str, Any]
    n_frames: int = 1
    requires: Sequence[str] = Field(default_factory=list)


class PipeSpec(ConfigYAMLMixin):
    buff: dict[str, Any]
    prep: dict[str, NodeSpec]
    init: dict[str, NodeSpec]
    iter: dict[str, NodeSpec]


class MovieSpec(BaseModel):
    stream_url: str


class PlotSpec(BaseModel):
    width: int | str
    height: int | str
    max_points: int


class GUISpec(BaseModel):
    prep_movie: MovieSpec
    metric_plot: PlotSpec
    footprint_movie: MovieSpec
