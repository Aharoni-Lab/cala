from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, TypedDict

import xarray as xr
from river.base.transformer import BaseTransformer


@dataclass
class Frame:
    array: xr.DataArray
    index: int
    timestamp: datetime | None = None


class Preprocessor(Enum):
    DOWNSAMPLE = "downsample"
    DENOISE = "denoise"
    GLOW_REMOVAL = "glow_removal"
    BACKGROUND_REMOVAL = "background_removal"
    RIGID_STABILIZATION = "rigid_stabilization"

    def get_class(self) -> type[BaseTransformer]:
        from cala.streaming.preprocess import (
            BackgroundEraser,
            Denoiser,
            Downsampler,
            GlowRemover,
            RigidStabilizer,
        )

        return {
            Preprocessor.DOWNSAMPLE: Downsampler,
            Preprocessor.DENOISE: Denoiser,
            Preprocessor.GLOW_REMOVAL: GlowRemover,
            Preprocessor.BACKGROUND_REMOVAL: BackgroundEraser,
            Preprocessor.RIGID_STABILIZATION: RigidStabilizer,
        }[self]


class Initializer(Enum):
    FOOTPRINTS = "footprints"
    TRACES = "traces"
    COMPONENT_STATS = "component_stats"
    PIXEL_STATS = "pixel_stats"
    OVERLAPS = "overlaps"
    RESIDUALS = "residuals"

    def get_class(self) -> type[BaseTransformer]:
        from cala.streaming.init.common import FootprintsInitializer, TracesInitializer
        from cala.streaming.init.odl import (
            ComponentStatsInitializer,
            OverlapsInitializer,
            PixelStatsInitializer,
            ResidualInitializer,
        )

        return {
            Initializer.FOOTPRINTS: FootprintsInitializer,
            Initializer.TRACES: TracesInitializer,
            Initializer.COMPONENT_STATS: ComponentStatsInitializer,
            Initializer.PIXEL_STATS: PixelStatsInitializer,
            Initializer.OVERLAPS: OverlapsInitializer,
            Initializer.RESIDUALS: ResidualInitializer,
        }[self]


class Iterator(Enum):
    TRACES = "traces"
    COMPONENT_STATS = "component_stats"
    PIXEL_STATS = "pixel_stats"
    DETECT = "detect"
    FOOTPRINTS = "footprints"
    OVERLAPS = "overlaps"

    def get_class(self) -> type[BaseTransformer]:
        from cala.streaming.iterate import (
            ComponentStatsUpdater,
            Detector,
            FootprintsUpdater,
            OverlapsUpdater,
            PixelStatsUpdater,
            TracesUpdater,
        )

        return {
            Iterator.TRACES: TracesUpdater,
            Iterator.COMPONENT_STATS: ComponentStatsUpdater,
            Iterator.PIXEL_STATS: PixelStatsUpdater,
            Iterator.DETECT: Detector,
            Iterator.FOOTPRINTS: FootprintsUpdater,
            Iterator.OVERLAPS: OverlapsUpdater,
        }[self]


class PreprocessStep(TypedDict):
    transformer: type
    params: dict[str, Any]
    requires: NotRequired[Sequence[str]]


class InitializationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int  # Number of frames to use
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class IterationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class StreamingConfig(TypedDict):
    preprocess: dict[str, PreprocessStep]
    initialization: dict[str, InitializationStep]
    iteration: dict[str, IterationStep]
