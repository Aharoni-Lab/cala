from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NotRequired, TypedDict

import xarray as xr


@dataclass
class Frame:
    array: xr.DataArray
    index: int
    timestamp: datetime | None = None


class PreprocessStep(TypedDict):
    transformer: str
    params: dict[str, Any]
    requires: NotRequired[Sequence[str]]


class InitializationStep(TypedDict):
    transformer: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int  # Number of frames to use
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class IterationStep(TypedDict):
    transformer: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class StreamingConfig(TypedDict):
    preprocess: dict[str, PreprocessStep]
    initialization: dict[str, InitializationStep]
    iteration: dict[str, IterationStep]
