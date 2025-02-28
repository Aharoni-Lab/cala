from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Type, TypeVar, cast

import xarray as xr
from river.base import SupervisedTransformer, Transformer

from cala.streaming.core.components import ComponentManager
from cala.streaming.core.components.types import ComponentType


# Future imports:
# from .component_stats import ComponentStatsResult
# from .pixel_stats import PixelStatsResult
# from .residual_buffer import ResidualBufferResult


class InitializerType(Enum):
    """Types of initialization transformers."""

    SPATIAL = auto()
    TEMPORAL = auto()
    # Future types:
    # COMPONENT_STATS = auto()
    # PIXEL_STATS = auto()
    # RESIDUAL = auto()


T = TypeVar("T", Transformer, SupervisedTransformer)


@dataclass
class SpatialInitializationResult:
    """Result from spatial initialization."""

    background: xr.DataArray = field(init=False)
    neurons: xr.DataArray = field(init=False)


@dataclass
class TemporalInitializationResult:
    """Result from temporal initialization."""

    traces: xr.DataArray = field(init=False)


def manager_interface(initializer_type: InitializerType):
    """Decorator that adds ComponentManager interaction to initialization transformers.

    Args:
        initializer_type: Type of initialization being performed
    """

    def decorator(transformer_class: Type[T]) -> Type[T]:
        class ManagerWrappedTransformer(cast(type, transformer_class)):
            def learn_one(self, components: ComponentManager, X: xr.DataArray) -> T:
                """Learn step extracts needed data from manager and passes to transformer."""
                match initializer_type:
                    case InitializerType.SPATIAL:
                        args = (
                            (X, None)
                            if isinstance(self, SupervisedTransformer)
                            else (X,)
                        )
                        return super().learn_one(*args)
                    case InitializerType.TEMPORAL:
                        args = (
                            (components.footprints, X)
                            if isinstance(self, SupervisedTransformer)
                            else (X,)
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
                # Get appropriate input based on initializer type
                transform_one_input = self._get_transform_one_input(components)

                # Run transformer
                result = super().transform_one(transform_one_input)

                # Update manager based on result type
                self._update_manager(components, result)

                return components

            def _get_transform_one_input(self, components: ComponentManager):
                """Get input needed by this type of transformer."""
                match initializer_type:
                    case InitializerType.SPATIAL:
                        return None
                    case InitializerType.TEMPORAL:
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
                    case SpatialInitializationResult():
                        components.populate_from_footprints(
                            result.background, ComponentType.BACKGROUND
                        )
                        components.populate_from_footprints(
                            result.neurons, ComponentType.NEURON
                        )
                    case TemporalInitializationResult():
                        components.populate_from_traces(result.traces)
                    # Future cases:
                    # case ComponentStatsResult():
                    #     components.update_component_stats(result.stats)
                    # case PixelStatsResult():
                    #     components.update_pixel_stats(result.stats)
                    # case ResidualBufferResult():
                    #     components.update_residuals(result.residuals)

        ManagerWrappedTransformer.__name__ = transformer_class.__name__
        return ManagerWrappedTransformer  # type: ignore

    return decorator
