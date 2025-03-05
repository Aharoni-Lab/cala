import inspect
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import TypedDict, Callable, Any, NotRequired, Dict, Sequence, List

import networkx as nx
import xarray as xr


class Frame(xr.DataArray):
    pass


class Frames(List[Frame]):
    pass


class Footprints(xr.DataArray):
    pass


class Traces(xr.DataArray):
    pass


class NeuronFootprints(Footprints):
    pass


class NeuronTraces(Traces):
    pass


class BackgroundFootprints(Footprints):
    pass


class BackgroundTraces(Traces):
    pass


# Example config
# config = {
#     "initialization": {
#         "motion_correction": {
#             "transformer": MotionCorrection,
#             "params": {
#                 "max_shift": 10,
#                 "patch_size": 50
#             }
#         },
#         "neuron_detection": {
#             "transformer": CNMFDetection,
#             "params": {
#                 "num_components": 100,
#                 "merge_threshold": 0.85
#             },
#             "requires": ["motion_corrected_frames"]  # Dependencies from previous steps
#         },
#         "trace_extraction": {
#             "transformer": TraceExtractor,
#             "params": {
#                 "method": "pca",
#                 "components": 5
#             },
#             "requires": ["neuron_footprints"]
#         }
#     }
# }


class InitializationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class Config(TypedDict):
    initialization: dict[str, InitializationStep]


class TransformerMeta(ABCMeta):
    """Metaclass for streaming transformers that extracts method signatures."""

    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)

        # Extract signatures from methods
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                setattr(attr_value, "__signature__", cls._extract_signature(attr_value))

        return new_cls

    @staticmethod
    def _extract_signature(func: Callable) -> dict[str, type]:
        """
        Extracts parameter and return type hints from a callable.

        Args:
            func: The callable to analyze

        Returns:
            Dictionary mapping parameter names to their type hints
        """

        # Get function signature
        signature = inspect.signature(func)

        # Extract parameter types (excluding 'self' if present)
        param_types = {
            name: param.annotation
            for name, param in signature.parameters.items()
            if name != "self"
        }

        # Add return type if specified
        if signature.return_annotation:
            param_types["return"] = signature.return_annotation

        return param_types


@dataclass
class InitState:
    neuron_footprints: NeuronFootprints = field(
        default_factory=lambda: NeuronFootprints()
    )
    neuron_traces: NeuronTraces = field(default_factory=lambda: NeuronTraces())
    background_footprints: BackgroundFootprints = field(
        default_factory=lambda: BackgroundFootprints()
    )
    background_traces: BackgroundTraces = field(
        default_factory=lambda: BackgroundTraces()
    )

    @property
    def all_footprints(self) -> Footprints:
        return Footprints(
            xr.concat([self.neuron_footprints, self.background_footprints])
        )

    @property
    def all_traces(self) -> Traces:
        return Traces(xr.concat([self.neuron_traces, self.background_traces]))

    def update(self, result: xr.DataArray | tuple[xr.DataArray, ...]):
        """Update state with latest processing results.

        Args:
            result: Either a single DataArray or tuple of DataArrays to update state with
        """
        results = (result,) if isinstance(result, xr.DataArray) else result

        # Get mapping of types to attribute names
        type_map = {
            annotated_type: attr
            for attr, annotated_type in inspect.get_annotations(self.__init__).items()
            if annotated_type is not None
        }

        # Update attributes whose types match result values
        for value in results:
            if attr_name := type_map.get(type(value)):
                setattr(self, attr_name, value)


@dataclass
class Runner:
    config: Config
    is_initialized: bool = False

    def _get_injects(self, state: InitState, function: Callable) -> Dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results
            function: function to get signature from

        Returns:
            Dictionary mapping parameter names to matching state values
        """
        # Get mapping of state attribute types
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

        # Initialize state
        state = InitState()

        # Execute transformers in order
        for step in execution_order:
            config = self.config["initialization"][step]
            params = config.get("params", {})
            transformer = config["transformer"](**params)

            # Get dependencies by matching signature types
            learn_injects = self._get_injects(state, transformer.learn_one)
            transform_injects = self._get_injects(state, transformer.transform_one)

            # Initialize and run transformer
            transformer.learn_one(frame=frame, **learn_injects)
            result = transformer.transform_one(**transform_injects)

            state.update(result)

        self.is_initialized = True
        return state

    def update(self): ...
