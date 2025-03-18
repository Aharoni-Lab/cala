from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Literal

import networkx as nx
import xarray as xr
from river import compose

from cala.streaming.composer.pipe_config import StreamingConfig
from cala.streaming.core import Parameters, Distributor
from cala.streaming.util.buffer import Buffer


@dataclass
class Runner:
    config: StreamingConfig
    _buffer: Buffer = field(init=False)
    _state: Distributor = field(default_factory=lambda: Distributor())
    is_initialized: bool = False

    def __post_init__(self):
        self._buffer = Buffer(
            buffer_size=10,
        )

    def preprocess(self, frame: xr.DataArray) -> Dict[str, Any]:
        execution_order = self._create_dependency_graph(self.config["preprocess"])

        pipeline = compose.Pipeline()

        for step in execution_order:
            transformer = self._build_transformer(process="preprocess", step=step)

            pipeline = pipeline | transformer

        pipeline.learn_one(x=frame)
        result = pipeline.transform_one(x=frame)

        return result

    def initialize(self, frame: xr.DataArray):
        """Initialize transformers in dependency order."""
        self._buffer.add_frame(frame)

        execution_order = self._create_dependency_graph(self.config["initialization"])
        status = [False] * len(execution_order)

        for idx, step in enumerate(execution_order):
            if status[idx]:
                continue

            n_frames = self.config["initialization"][step].get("n_frames", 1)
            if not self._buffer.is_ready(n_frames):
                break

            transformer = self._build_transformer(process="initialization", step=step)
            result = self._learn_transform(
                transformer=transformer, frame=self._buffer.get_latest(n_frames)
            )
            if result is not None:
                status[idx] = True

            self._state.collect(result)

        if all(status):
            self.is_initialized = True

    def extract(self, frame: xr.DataArray):
        execution_order = self._create_dependency_graph(self.config["extraction"])

        # Execute transformers in order
        for step in execution_order:
            transformer = self._build_transformer(process="extraction", step=step)
            result = self._learn_transform(transformer=transformer, frame=frame)

            self._state.collect(result)

    def _build_transformer(
        self, process: Literal["preprocess", "initialization", "extraction"], step: str
    ):
        config = self.config[process][step]
        params = config.get("params", {})
        transformer = config["transformer"]

        param_class = next(
            (
                type_
                for type_ in transformer.__annotations__.values()
                if issubclass(type_, Parameters)
            ),
            None,
        )
        if param_class:
            param_obj = param_class(**params)
            transformer = transformer(param_obj)
        else:
            transformer = transformer()

        return transformer

    def _learn_transform(self, transformer, frame: xr.DataArray):
        # Get dependencies by matching signature categories
        learn_injects = self._get_injects(self._state, transformer.learn_one)
        transform_injects = self._get_injects(self._state, transformer.transform_one)

        # Initialize and run transformer
        transformer.learn_one(frame=frame, **learn_injects)
        result = transformer.transform_one(**transform_injects)

        return result

    @staticmethod
    def _get_injects(state: Distributor, function: Callable) -> Dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results
            function: function to get signature from

        Returns:
            Dictionary mapping parameter names to matching state values
        """
        # Ask data exchange for the type matching value
        matches = {}
        for param_name, param_type in function.__signature__.items():
            if param_name == "return":
                continue
            value = state.get(param_type)
            if value is not None:
                matches[param_name] = value

        return matches

    @staticmethod
    def _create_dependency_graph(steps: dict) -> list:
        # Create dependency graph
        graph = nx.DiGraph()

        for step in steps:
            graph.add_node(step)

        for step, config in steps.items():
            if "requires" in config:
                for dep in config["requires"]:
                    graph.add_edge(dep, step)

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Transformer dependencies contain cycles")

        return list(nx.topological_sort(graph))
