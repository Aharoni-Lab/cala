from typing import Any

from noob.types import AbsoluteIdentifier
from pydantic import BaseModel, field_validator, ConfigDict


class Grid(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: AbsoluteIdentifier
    params: dict[str, Any] | None = None


class GUISpec(BaseModel):
    grids: dict[str, Grid]

    @field_validator("grids", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `grids` dictionary into the grid config
        """
        assert isinstance(value, dict)
        for id_, grid in value.items():
            if "id" not in grid:
                grid["id"] = id_
        return value
