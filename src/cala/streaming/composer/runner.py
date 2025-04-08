import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import (
    Any,
    Literal,
    get_type_hints,
)

import xarray as xr
from river import compose

from cala.config import Frame, StreamingConfig
from cala.streaming.core import Parameters
from cala.streaming.core.distribution import Distributor
from cala.streaming.nodes import Node
from cala.streaming.util.buffer import Buffer

logger = logging.getLogger(__name__)


@dataclass
class Runner:
    """Manages the execution of streaming calcium imaging analysis pipeline.

    This class orchestrates the preprocessing, initialization, and iterate steps
    of the calcium imaging analysis pipeline according to a provided configuration.
    """

    config: StreamingConfig
    """Configuration defining the pipeline structure and parameters."""
    _buffer: Buffer = field(init=False)
    """Internal frame buffer for multi-frame operations."""
    _state: Distributor = field(default_factory=lambda: Distributor())
    """Current state of the pipeline containing computed results."""
    execution_order: list[str] | None = None
    """Ordered list of initialization steps."""
    _init_statuses: list[bool] | None = None
    """Completion status for each initialization step."""
    is_initialized: bool = False
    """Whether the pipeline initialization is complete."""

    def __post_init__(self) -> None:
        """Initialize the frame buffer after instance creation."""
        self._buffer = Buffer(
            buffer_size=10,
        )

    def preprocess(self, frame: Frame) -> Frame:
        """Execute preprocessing steps on a single frame.

        Args:
            frame: Input frame to preprocess.

        Returns:
            Dictionary containing preprocessed results.
        """
        execution_order = self._create_dependency_graph(self.config.preprocess)

        pipeline = compose.Pipeline()

        for step in execution_order:
            transformer = self._build_transformer(process="preprocess", step=step)

            pipeline = pipeline | transformer

        pipeline.learn_one(x=frame.array)
        result = pipeline.transform_one(x=frame.array)

        frame.array = result
        return frame

    def initialize(self, frame: Frame) -> None:
        """Initialize pipeline transformers in dependency order.

        Executes initialization steps that may require multiple frames. Steps are executed
        in topological order based on their dependencies.

        Args:
            frame: New frame to use for initialization.
        """
        self._buffer.add_frame(frame.array)

        if not self.execution_order or not self._init_statuses:
            self.execution_order = self._create_dependency_graph(self.config.initialization)
            self._init_statuses = [False] * len(self.execution_order)

        for idx, step in enumerate(self.execution_order):
            if self._init_statuses[idx]:
                continue

            n_frames = getattr(self.config.initialization[step], "n_frames", 1)
            if not self._buffer.is_ready(n_frames):
                break

            transformer = self._build_transformer(process="initialization", step=step)
            result = self._learn_transform(
                transformer=transformer, frame=self._buffer.get_latest(n_frames)
            )
            if result is not None:
                self._init_statuses[idx] = True

            result_type = get_type_hints(transformer.transform_one, include_extras=True)["return"]
            self._state.init(result, result_type)

        if all(self._init_statuses):
            self.is_initialized = True

    def iterate(self, frame: Frame) -> None:
        """Execute iterate steps on a single frame.

        Args:
            frame: Input frame to process for component iterate.
        """
        execution_order = self._create_dependency_graph(self.config.iteration)
        logger.info(f"Footprint Count: {self._state.footprintstore.warehouse.sizes}")

        # Execute transformers in order
        for step in execution_order:
            logger.info(f"Executing step: {step}")
            transformer = self._build_transformer(process="iteration", step=step)
            result = self._learn_transform(transformer=transformer, frame=frame)

            result_type = get_type_hints(transformer.transform_one, include_extras=True)["return"]

            self._state.update(result, result_type)

    def _build_transformer(
        self, process: Literal["preprocess", "initialization", "iteration"], step: str
    ) -> Any:
        """Construct a transformer instance with configured parameters.

        Args:
            process: Type of process the transformer belongs to.
            step: Name of the configuration step.

        Returns:
            Configured transformer instance.
        """
        config = getattr(self.config, process)[step]
        params = getattr(config, "params", {})
        transformer = Node[config.transformer].value

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

    def _learn_transform(
        self, transformer: Any, frame: Frame
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Execute learn and transform steps for a transformer.

        Args:
            transformer: Transformer instance to execute.
            frame: Input frame to process.

        Returns:
            Transformation results.
        """
        learn_injects = self._get_injects(self._state, transformer.learn_one)
        transform_injects = self._get_injects(self._state, transformer.transform_one)

        # Initialize and run transformer
        transformer.learn_one(frame=frame, **learn_injects)
        result = transformer.transform_one(**transform_injects)

        return result

    @staticmethod
    def _get_injects(state: Distributor, function: Callable) -> dict[str, Any]:
        """Extract required dependencies from the current state based on function signature.

        Args:
            state: Current pipeline state containing all computed results.
            function: Function to get signature from.

        Returns:
            Dictionary mapping parameter names to matching state values.
        """
        matches = {}
        for param_name, param_type in get_type_hints(function, include_extras=True).items():
            if param_name == "return":
                continue

            value = state.get(param_type)
            if value is not None:
                matches[param_name] = value

        return matches

    @staticmethod
    def _create_dependency_graph(steps: dict) -> list[str]:
        """Create and validate a dependency graph for execution ordering.

        Args:
            steps: Dictionary of pipeline steps and their configurations.

        Returns:
            List of steps in topological order.

        Raises:
            ValueError: If dependencies contain cycles.
        """
        graph: dict[str, set[str]] = {}
        for step in steps:
            graph[step] = set()

        for step, config in steps.items():
            if "requires" in config:
                graph[step] = set(config["requires"])

        # Create and prepare the sorter
        ts = TopologicalSorter(graph)
        return list(ts.static_order())
