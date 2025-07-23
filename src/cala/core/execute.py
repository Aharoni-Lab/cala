import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, get_type_hints

import xarray as xr
from river import compose

from cala.core.distribute import Distributor

# from cala.gui.nodes import (
#     ComponentCounter,
#     ComponentCounterParams,
#     ComponentStreamer,
#     ComponentStreamerParams,
#     FrameCounter,
#     FrameCounterParams,
#     FrameStreamer,
#     FrameStreamerParams,
# )
from cala.logging import init_logger
from cala.models.params import Params
from cala.models.spec import NodeSpec, Pipe
from cala.util.buffer import Buffer

logger = init_logger(__name__)


@dataclass
class Executor:
    """Manages the execution of streaming image analysis pipeline.

    This class orchestrates the steps of the imaging analysis pipeline
    according to a provided configuration.
    """

    pipeline: Pipe
    """Configuration defining the pipeline structure and parameters."""

    _buffer: Buffer = field(init=False)
    """Internal frame buffer for multi-frame operations."""
    _state: Distributor = field(default_factory=lambda: Distributor())
    """Current state of the pipeline containing computed results."""
    _transformers: dict[str, Any] = field(default_factory=dict)
    """Cache of transformer instances for reuse."""
    execution_order: list[str] | None = None
    """Ordered list of initialization steps."""
    _init_statuses: list[bool] | None = None
    """Completion status for each initialization step."""
    is_initialized: bool = False
    """Whether the pipeline initialization is complete."""

    def __post_init__(self) -> None:
        """Initialize the frame buffer after instance creation."""
        # self.prep_movie_streamer = FrameStreamer(
        #     FrameStreamerParams(
        #         frame_rate=30,
        #         stream_dir=self.pipeline.output_dir / "prep_movie",
        #         segment_filename="stream%d.ts",
        #     )
        # )
        # self.frame_counter = FrameCounter(FrameCounterParams())
        # self.component_counter = ComponentCounter(ComponentCounterParams())
        # self.component_streamer = ComponentStreamer(
        #     ComponentStreamerParams(
        #         frame_rate=30,
        #         stream_dir=self.pipeline.output_dir / "components",
        #         segment_filename="stream%d.ts",
        #     )
        # )

        self._buffer = Buffer(
            buffer_size=self.pipeline.pipeline.buff["buffer_size"],
        )

    def preprocess(self, frame: xr.DataArray) -> xr.DataArray:
        """Execute preprocessing steps on a single frame.

        Args:
            frame: Input frame to preprocess.

        Returns:
            Dictionary containing preprocessed results.
        """
        execution_order = self._create_dependency_graph(self.pipeline.prep)

        pipeline = compose.Pipeline()

        # if self.pipeline.gui:
        #     self.frame_counter.learn_one(frame=frame)
        #     self.frame_counter.transform_one(_=frame)

        for step in execution_order:
            transformer = self._build_transformer(process="preprocess", step=step)

            pipeline = pipeline | transformer

        pipeline.learn_one(x=frame)
        result = pipeline.transform_one(x=frame)

        # if self.pipeline.gui:
        #     # plug in prep_movie_display
        #     self.prep_movie_streamer.learn_one(frame=frame)
        #     self.prep_movie_streamer.transform_one(frame=result)

        frame = result

        return frame

    def initialize(self, frame: xr.DataArray) -> None:
        """Initialize pipeline transformers in dependency order.

        Executes initialization steps that may require multiple frames. Steps are executed
        in topological order based on their dependencies.

        Args:
            frame: New frame to use for initialization.
        """
        self._buffer.add_frame(frame)

        if not self.execution_order or not self._init_statuses:
            self.execution_order = self._create_dependency_graph(self.pipeline.init)
            self._init_statuses = [False] * len(self.execution_order)

        for idx, step in enumerate(self.execution_order):
            if self._init_statuses[idx]:
                continue

            n_frames = getattr(self.pipeline.init[step], "n_frames", 1)
            if not self._buffer.is_ready(n_frames):
                break

            transformer = self._build_transformer(process="initialization", step=step)
            result = self._learn_transform(
                transformer=transformer, frame=self._buffer.get_latest(n_frames)
            )
            if result is not None:
                self._init_statuses[idx] = True

            result_type = get_type_hints(transformer.transform_one, include_extras=True)["return"]
            self._state.init(
                result,
                result_type,
                self.pipeline.buff["buffer_size"],
                self.pipeline.output_dir,
            )

        if all(self._init_statuses):
            self.is_initialized = True

    def iterate(self, frame: xr.DataArray) -> None:
        """Execute iterate steps on a single frame.

        Args:
            frame: Input frame to process for component iterate.
        """
        execution_order = self._create_dependency_graph(self.pipeline.iter)

        # Execute transformers in order
        for step in execution_order:
            logger.info(f"Executing step: {step}")
            transformer = self._build_transformer(process="iteration", step=step)
            result = self._learn_transform(transformer=transformer, frame=frame)

            result_type = get_type_hints(transformer.transform_one, include_extras=True)["return"]

            self._state.update(result, result_type)

        if getattr(self._state, "footprintstore", None):
            self.component_counter.learn_one(self._state.footprintstore)
            self.component_counter.transform_one()

            self.component_streamer.learn_one(self._state.footprintstore)
            self.component_counter.transform_one()

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
        config = getattr(self.pipeline, process)[step]
        if self._transformers.get(config.id, None):
            return self._transformers[config.id]

        params = getattr(config, "params", {})
        module_name, class_name = config.id.rsplit(".", 1)
        transformer = getattr(importlib.import_module(module_name), class_name)

        param_class = next(
            (type_ for type_ in transformer.__annotations__.values() if issubclass(type_, Params)),
            None,
        )
        if param_class:
            param_obj = param_class(**params)
            transformer = transformer(param_obj)
        else:
            transformer = transformer()

        self._transformers[config.id] = transformer

        return transformer

    def _learn_transform(
        self, transformer: Any, frame: xr.DataArray
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

    def cleanup(self) -> None:
        """
        1. Store cleanup (e.g., closing Zarr files)
        2. Clear internal buffer to release memory
        3. Clear state variables
        """
        ...
