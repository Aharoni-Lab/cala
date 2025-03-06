from dataclasses import dataclass
from typing import Callable, Any, Dict

import networkx as nx

from cala.streaming.core.components import DataOutlet
from cala.streaming.pipe_config import StreamingConfig


@dataclass
class Runner:
    config: StreamingConfig
    state = DataOutlet()
    is_initialized: bool = False

    def _get_injects(self, state: DataOutlet, function: Callable) -> Dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results
            function: function to get signature from

        Returns:
            Dictionary mapping parameter names to matching state values
        """
        # Get mapping of state attribute categories
        state_types = {
            type(getattr(state, attr)): (attr, getattr(state, attr))
            for attr in vars(state)
            if getattr(state, attr) is not None
        }

        # Match function parameters with state attributes by type
        matches = {}
        for param_name, param_type in function.__signature__.items():
            if param_name == "return":
                continue
            if param_type in state_types:
                _, value = state_types[param_type]
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

    def initialize(self, frame: Frame):
        """Initialize transformers in dependency order."""
        execution_order = self._create_dependency_graph(self.config["initialization"])

        # Execute transformers in order
        for step in execution_order:
            config = self.config["initialization"][step]
            params = config.get("params", {})
            transformer = config["transformer"](**params)

            # Get dependencies by matching signature categories
            learn_injects = self._get_injects(self.state, transformer.learn_one)
            transform_injects = self._get_injects(self.state, transformer.transform_one)

            # Initialize and run transformer
            transformer.learn_one(frame=frame, **learn_injects)
            result = transformer.transform_one(**transform_injects)

            self.state.update(result)

        self.is_initialized = True
        return self.state

    def update(self): ...
