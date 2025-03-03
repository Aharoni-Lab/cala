from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Type, TypeVar, cast, Self

import xarray as xr
import numpy as np
from river.base import SupervisedTransformer, Transformer

from cala.streaming.core.components import ComponentManager
from cala.streaming.core.components.types import ComponentType
from cala.streaming.data.buffer import RingBuffer


class InitializerType(Enum):
    """Types of initialization transformers."""

    FOOTPRINTS = auto()
    TRACES = auto()
    # Future types:
    # COMPONENT_STATS = auto()
    # PIXEL_STATS = auto()
    # RESIDUAL = auto()


T = TypeVar("T", Transformer, SupervisedTransformer)


@dataclass
class FootprintsInitializationResult:
    """Result from spatial initialization."""

    background: xr.DataArray = field(init=False)
    neurons: xr.DataArray = field(init=False)


@dataclass
class TracesInitializationResult:
    """Result from temporal initialization."""

    traces: xr.DataArray = field(init=False)


@dataclass
class InitializerBuffer:
    """Wrapper for initialization transformers that need frame buffering."""

    min_frames_required: int
    """Minimum number of frames required before processing"""
    frames_axis: str
    """Name of the frames axis"""
    frame_buffer: RingBuffer = field(init=False)
    """Buffer to store frames"""
    is_buffer_initialized: bool = field(default=False, init=False)
    """Whether the buffer has been initialized"""
    frame_dims: tuple = field(init=False)
    """Frame dimensions"""
    frame_coords: dict = field(init=False)
    """Frame coordinates"""
    original_dtype: np.dtype = field(init=False)
    """Original dtype of the input frames"""

    def initialize_buffer(self, frame: xr.DataArray):
        """Initialize the frame buffer with dimensions from the xarray frame."""
        # Store frame dimensions and coordinates for later use
        self.frame_dims = tuple(name for name in frame.dims if name != self.frames_axis)
        self.frame_coords = {
            name: frame.coords[name].values for name in self.frame_dims
        }

        # Create buffer with appropriate shape
        frame_shape = tuple(frame.sizes[dim] for dim in self.frame_dims)
        self.frame_buffer = RingBuffer(
            buffer_size=self.min_frames_required,
            frame_shape=frame_shape,
            dtype=frame.dtype,
        )
        self.is_buffer_initialized = True

    def add_frame(self, frame: xr.DataArray) -> bool:
        """Add a frame to the buffer and return True if buffer is ready for processing."""
        if not self.is_buffer_initialized:
            self.initialize_buffer(frame)

        self.frame_buffer.add_frame(frame.values)
        return self.frame_buffer.total_frames >= self.min_frames_required

    def get_frames_xarray(self) -> xr.DataArray:
        """Get buffered frames as an xarray DataArray with original dimensions."""
        frames = self.frame_buffer.get_buffer_in_order()

        # Create coordinates dict including frames
        coords = {self.frames_axis: range(len(frames))}
        coords.update(self.frame_coords)

        # Create DataArray with original dimensions
        return xr.DataArray(
            frames,
            dims=(self.frames_axis,) + self.frame_dims,
            coords=coords,
        )


def manager_interface(initializer_type: InitializerType):
    """Decorator that adds ComponentManager interaction to initialization transformers.

    Args:
        initializer_type: Type of initialization being performed
    """

    def decorator(transformer_class: Type[T]) -> Type[T]:
        class ManagerWrappedTransformer(cast(type, transformer_class)):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.is_initialized = False
                self.buffer_ready = False

                # Initialize buffer based on initializer type and parameters
                match initializer_type:
                    case InitializerType.FOOTPRINTS:
                        self.buffer = InitializerBuffer(
                            min_frames_required=1,
                            frames_axis="frames",
                        )
                    case InitializerType.TRACES:
                        self.buffer = InitializerBuffer(
                            min_frames_required=self.params.num_frames_to_use,
                            frames_axis=self.params.frames_axis,
                        )

            def learn_one(self, components: ComponentManager, X: xr.DataArray) -> T:
                """Learn step extracts needed data from manager and passes to transformer."""
                # Add frame to buffer
                self.buffer_ready = self.buffer.add_frame(X)
                if not self.buffer_ready:
                    return cast(T, self)

                # Get buffered frames
                frames = self.buffer.get_frames_xarray()

                # Process based on initializer type
                match initializer_type:
                    case InitializerType.FOOTPRINTS:
                        args = (
                            (frames.isel(frames=-1), None)
                            if isinstance(self, SupervisedTransformer)
                            else (frames.isel(frames=-1),)
                        )
                        return super().learn_one(*args)
                    case InitializerType.TRACES:
                        args = (
                            (components.footprints, frames)
                            if isinstance(self, SupervisedTransformer)
                            else (frames,)
                        )
                        return super().learn_one(*args)
                    # Future cases:
                    # case InitializerType.COMPONENT_STATS:
                    #     super().learn_one(components.traces, X)
                    # case InitializerType.PIXEL_STATS:
                    #     super().learn_one(components.traces, X)
                    # case InitializerType.RESIDUAL:
                    #     super().learn_one(components.footprints, components.traces, X)
                return self

            def transform_one(self, components: ComponentManager) -> ComponentManager:
                """Transform step runs transformer and updates manager."""
                if not self.buffer_ready:
                    return components

                # Get appropriate input based on initializer type
                transform_one_input = self._get_transform_one_input(components)

                # Run transformer
                result = super().transform_one(transform_one_input)

                # Update manager based on result type
                self._update_manager(components, result)

                self.is_initialized = True
                return components

            def _get_transform_one_input(self, components: ComponentManager):
                """Get input needed by this type of transformer."""
                match initializer_type:
                    case InitializerType.FOOTPRINTS:
                        return None
                    case InitializerType.TRACES:
                        return components.footprints
                    # Future cases:
                    # case InitializerType.COMPONENT_STATS:
                    #     return components.traces
                    # case InitializerType.PIXEL_STATS:
                    #     return components.traces
                    # case InitializerType.RESIDUAL:
                    #     return (components.footprints, components.traces)

            def _update_manager(self, components: ComponentManager, result):
                """Update manager based on result type."""
                match result:
                    case FootprintsInitializationResult():
                        components.populate_from_footprints(
                            result.background, ComponentType.BACKGROUND
                        )
                        components.populate_from_footprints(
                            result.neurons, ComponentType.NEURON
                        )
                    case TracesInitializationResult():
                        components.populate_from_traces(result.traces)
                    # Future cases:
                    # case ComponentStatsResult():
                    #     components.update_component_stats(result.stats)
                    # case PixelStatsResult():
                    #     components.update_pixel_stats(result.stats)
                    # case ResidualBufferResult():
                    #     components.update_residuals(result.residuals)

            def learn_transform_one(
                self, components: ComponentManager, X: xr.DataArray
            ) -> ComponentManager:
                """Learn and transform step."""
                self.learn_one(components=components, X=X)
                return self.transform_one(components)

        ManagerWrappedTransformer.__name__ = transformer_class.__name__
        return ManagerWrappedTransformer  # type: ignore

    return decorator
