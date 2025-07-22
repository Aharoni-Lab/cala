from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from cala.config.yaml import ConfigYAMLMixin


class Node(BaseModel):
    id: str  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    n_frames: int = 1  # Number of frames to use
    requires: Sequence[str] = []  # Optional dependencies


class Pipeline(ConfigYAMLMixin):
    buff: dict[str, Any]
    prep: dict[str, Node]
    init: dict[str, Node]
    iter: dict[str, Node]
