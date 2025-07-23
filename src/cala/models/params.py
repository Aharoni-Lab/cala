import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from cala.models.axis import Axis


@dataclass
class Params(ABC, Axis):
    """Parameter management and validation"""

    def __post_init__(self) -> None:
        """Validate parameters after initialization"""
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Validate all parameters"""
        pass

    def to_dict(self) -> dict:
        """Convert parameters to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def save(self, filename: str) -> None:
        """Save parameters to file"""
        import json

        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Params":
        """Create parameters from dictionary"""
        return cls(**data)

    @classmethod
    def load(cls, filename: str) -> "Params":
        """Load parameters from file"""
        with open(filename) as f:
            return cls.from_dict(json.load(f))

    def copy(self) -> "Params":
        """Create a deep copy of parameters"""
        return deepcopy(self)

    def update(self, **kwargs: Any) -> "Params":
        """Create new parameters with updated values"""
        return replace(self, **kwargs)

    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = ["Parameters:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
