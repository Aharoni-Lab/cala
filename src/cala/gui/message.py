from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FrameIndex(BaseModel):
    type_: Literal["frame_index"]
    index: int


class ComponentCount(BaseModel):
    type_: Literal["component_count"]
    index: int
    count: int


Payload = FrameIndex | ComponentCount


class WebsocketMessage(BaseModel):
    id: str = ""
    payload: Payload = Field(discriminator="type_")
    timestamp: datetime = Field(default_factory=datetime.now)
    error: str | None = None
