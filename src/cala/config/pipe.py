from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import xarray as xr
from pydantic import BaseModel


@dataclass
class Frame:
    array: xr.DataArray
    index: int
    timestamp: datetime | None = None


class PreprocessStep(BaseModel):
    transformer: str
    params: dict[str, Any]
    requires: Sequence[str] = []


class InitializationStep(BaseModel):
    transformer: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int = 1  # Number of frames to use
    requires: Sequence[str] = []  # Optional dependencies


class IterationStep(BaseModel):
    transformer: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: Sequence[str] = []  # Optional dependencies


class StreamingConfig(BaseModel):
    preprocess: dict[str, PreprocessStep]
    initialization: dict[str, InitializationStep]
    iteration: dict[str, IterationStep]
