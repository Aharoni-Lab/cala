from typing import Any

from noob import Tube
from noob.tube import TubeSpecification
from noob.types import ConfigSource
from noob.yaml import ConfigYAMLMixin
from pydantic import BaseModel, Field


class PipeSpec(ConfigYAMLMixin):
    buff: dict[str, Any] = Field(default_factory=dict)
    prep: TubeSpecification = Field(default_factory=TubeSpecification)
    init: TubeSpecification = Field(default_factory=TubeSpecification)
    iter: TubeSpecification = Field(default_factory=TubeSpecification)


class Pipe(BaseModel):
    buff: dict[str, Any] = Field(default_factory=dict)
    prep: Tube
    init: Tube
    iter: Tube

    @classmethod
    def from_specification(cls, spec: PipeSpec | ConfigSource) -> "Pipe":
        """
        Instantiate a tube model from its configuration

        Args:
            spec (TubeSpecification): the tube config to instantiate
        """
        spec = PipeSpec.from_any(spec)

        tubes = cls._init_tubes(spec)

        return cls(buff=spec.buff, prep=tubes["prep"], init=tubes["init"], iter=tubes["iter"])

    @classmethod
    def _init_tubes(cls, spec: PipeSpec) -> dict[str, Tube]:
        return {
            "prep": Tube.from_specification(spec.prep),
            "init": Tube.from_specification(spec.init),
            "iter": Tube.from_specification(spec.iter),
        }


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
