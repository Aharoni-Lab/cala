from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

import xarray as xr
from noob import SynchronousRunner

from cala.core.distro import Distro

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
from cala.models import Frame
from cala.models.spec import Pipe
from cala.util.buffer import Buffer

logger = init_logger(__name__)


@dataclass
class Executor:
    """Manages the execution of streaming image analysis pipeline.

    This class orchestrates the steps of the imaging analysis pipeline
    according to a provided configuration.
    """

    pipe: Pipe
    """Configuration defining the pipeline structure and parameters."""
    _buffer: Buffer = field(init=False)
    """Internal frame buffer for multi-frame operations."""
    store: Distro = field(default_factory=lambda: Distro())
    """Current state of the pipeline containing computed results."""
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
            size=self.pipe.buff["size"],
        )

    def preprocess(self) -> Frame:
        """
        Execute preprocessing steps on a single frame.

        Returns:
            Preprocessed frame.
        """
        runner = SynchronousRunner(self.pipe.prep)
        frame = runner.process()

        # if self.pipeline.gui:
        #     self.frame_counter.learn_one(frame=frame)
        #     self.frame_counter.transform_one(_=frame)

        # if self.pipeline.gui:
        #     # plug in prep_movie_display
        #     self.prep_movie_streamer.learn_one(frame=frame)
        #     self.prep_movie_streamer.transform_one(frame=result)

        return frame

    def initialize(self) -> None:
        """
        Initialize pipeline transformers in dependency order.

        Executes initialization steps that may require multiple frames. Steps are executed
        in topological order based on their dependencies.
        """
        # self._buffer.add_frame(frame)

        if not self.execution_order or not self._init_statuses:
            self.execution_order = self._create_dependency_graph(self.pipe.init)
            self._init_statuses = [False] * len(self.execution_order)

        for idx, step in enumerate(self.execution_order):
            if self._init_statuses[idx]:
                continue

            n_frames = getattr(self.pipe.init[step], "n_frames", 1)
            if not self._buffer.is_ready(n_frames):
                break

            transformer = self._build_transformer(process="initialization", step=step)
            result = self.process(node=transformer, frame=self._buffer.get_latest(n_frames))
            if result is not None:
                self._init_statuses[idx] = True

            result_type = get_type_hints(transformer.transform_one, include_extras=True)["return"]
            self.store.init(
                result,
                result_type,
                self.pipe.buff["buffer_size"],
                self.pipe.output_dir,
            )

        if all(self._init_statuses):
            self.is_initialized = True

    def iterate(self) -> None:
        """Execute iterate steps on a single frame."""

        # if getattr(self._state, "footprintstore", None):
        #     self.component_counter.learn_one(self._state.footprintstore)
        #     self.component_counter.transform_one()
        #
        #     self.component_streamer.learn_one(self._state.footprintstore)
        #     self.component_counter.transform_one()

    def process(self, node: Any, frame: xr.DataArray) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Execute learn and transform steps for a transformer.

        Args:
            node: Transformer instance to execute.
            frame: Input frame to process.

        Returns:
            Transformation results.
        """
        learn_injects = self._get_injects(self.store, node.learn_one)
        transform_injects = self._get_injects(self.store, node.transform_one)

        # Initialize and run transformer
        node.learn_one(frame=frame, **learn_injects)
        result = node.transform_one(**transform_injects)

        return result

    @staticmethod
    def _get_injects(state: Distro, function: Callable) -> dict[str, Any]:
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
