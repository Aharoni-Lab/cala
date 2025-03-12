from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Literal

import networkx as nx
import xarray as xr
from river import compose

from cala.streaming.core import DataExchange
from cala.streaming.pipe_config import StreamingConfig


@dataclass
class Runner:
    config: StreamingConfig
    state: DataExchange = field(default_factory=lambda: DataExchange())
    is_initialized: bool = False

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
        execution_order = self._create_dependency_graph(self.config["initialization"])

        # Execute transformers in order
        for step in execution_order:
            transformer = self._build_transformer(process="initialization", step=step)

            result = self._learn_transform(transformer=transformer, frame=frame)

            self.state.collect(result)

        self.is_initialized = True

    def extract(self, frame: xr.DataArray):
        execution_order = self._create_dependency_graph(self.config["extraction"])

        # Execute transformers in order
        for step in execution_order:
            transformer = self._build_transformer(process="extraction", step=step)

            result = self._learn_transform(transformer=transformer, frame=frame)

            self.state.collect(result)

    def _build_transformer(
        self, process: Literal["preprocess", "initialization", "extraction"], step: str
    ):
        config = self.config[process][step]
        params = config.get("params", {})
        transformer = config["transformer"]

        param_class = transformer.__annotations__.get("params")
        if param_class:
            param_obj = param_class(**params)
            transformer = transformer(param_obj)
        else:
            transformer = transformer()

        return transformer

    def _learn_transform(self, transformer, frame: xr.DataArray):
        # Get dependencies by matching signature categories
        learn_injects = self._get_injects(self.state, transformer.learn_one)
        transform_injects = self._get_injects(self.state, transformer.transform_one)

        # Initialize and run transformer
        transformer.learn_one(frame=frame, **learn_injects)
        result = transformer.transform_one(**transform_injects)

        return result

    @staticmethod
    def _get_injects(state: DataExchange, function: Callable) -> Dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results
            function: function to get signature from

        Returns:
            Dictionary mapping parameter names to matching state values
        """
        # Get mapping of state attribute categories
        state_types = state.type_to_store

        # Match function parameters with state attributes by type
        matches = {}
        for param_name, param_type in function.__signature__.items():
            if param_name == "return":
                continue
            if param_type in state_types:
                value = getattr(state, state_types[param_type]).warehouse
                matches[param_name] = value
            elif getattr(param_type, "__bases__", None):
                try:
                    value = state.get_observable_x_component(param_type)
                    matches[param_name] = value
                except TypeError:
                    continue

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
