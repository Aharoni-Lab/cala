from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


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
