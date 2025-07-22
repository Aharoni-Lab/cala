import inspect
from abc import abstractmethod
from collections.abc import Sequence
from graphlib import TopologicalSorter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cala.config.yaml import ConfigYAMLMixin
from cala.models.config import (
    AbsoluteIdentifier,
    ConfigSource,
    PythonIdentifier,
    resolve_python_identifier,
)


class NodeSpec(BaseModel):
    id: str
    type_: AbsoluteIdentifier = Field(..., alias="type")
    params: dict[str, Any]
    n_frames: int = 1
    requires: Sequence[str] = Field(default_factory=list)


class PipeSpec(ConfigYAMLMixin):
    buff: dict[str, Any] = Field(default_factory=dict)
    prep: dict[str, NodeSpec] = Field(default_factory=dict)
    init: dict[str, NodeSpec] = Field(default_factory=dict)
    iter: dict[str, NodeSpec] = Field(default_factory=dict)

    @field_validator("prep", "init", "iter", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `nodes` dictionary into the node config
        """
        assert isinstance(value, dict)
        for id_, node in value.items():
            if "id" not in node:
                node["id"] = id_
        return value


class Node(BaseModel):
    id: str
    spec: NodeSpec

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any | None:
        """Process some input, emitting it. See subclasses for details"""
        pass

    @classmethod
    def from_specification(cls, spec: NodeSpec) -> "Node":
        """
        Create a node from its spec

        - resolve the node type
        - if a function, wrap it in a node class
        - if a class, just instantiate it
        """
        obj = resolve_python_identifier(spec.type_)

        params = spec.params if spec.params is not None else {}

        # check if function by checking if callable -
        # Node classes do not have __call__ defined and thus should not be callable
        if inspect.isclass(obj):
            if issubclass(obj, Node):
                return obj(id=spec.id, spec=spec, **params)
            else:
                raise NotImplementedError("Handle wrapping classes")
        else:
            raise NotImplementedError("Handle wrapping functions")


class Pipe(BaseModel):
    prep: dict[str, Node] = Field(default_factory=dict)
    init: dict[str, Node] = Field(default_factory=dict)
    iter: dict[str, Node] = Field(default_factory=dict)

    def graph(self) -> TopologicalSorter: ...

    @classmethod
    def from_specification(cls, spec: PipeSpec | ConfigSource) -> "Pipe":
        """
        Instantiate a tube model from its configuration

        Args:
            spec (TubeSpecification): the tube config to instantiate
        """
        spec = PipeSpec.from_any(spec)

        nodes = cls._init_nodes(spec)

        return cls(prep=nodes[0], init=nodes[1], iter=nodes[2])

    @classmethod
    def _init_nodes(
        cls, specs: PipeSpec
    ) -> tuple[
        dict[PythonIdentifier, Node], dict[PythonIdentifier, Node], dict[PythonIdentifier, Node]
    ]:
        prep_nodes = {spec.id: Node.from_specification(spec) for spec in specs.prep.values()}
        init_nodes = {spec.id: Node.from_specification(spec) for spec in specs.init.values()}
        iter_nodes = {spec.id: Node.from_specification(spec) for spec in specs.iter.values()}
        return prep_nodes, init_nodes, iter_nodes


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
