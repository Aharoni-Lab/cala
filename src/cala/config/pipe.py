from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from cala.config.yaml import ConfigYAMLMixin


class NodeSpec(BaseModel):
    id: str
    params: dict[str, Any]
    n_frames: int = 1
    requires: Sequence[str] = Field(default_factory=list)


class PipeSpec(ConfigYAMLMixin):
    buff: dict[str, Any]
    prep: dict[str, NodeSpec]
    init: dict[str, NodeSpec]
    iter: dict[str, NodeSpec]
