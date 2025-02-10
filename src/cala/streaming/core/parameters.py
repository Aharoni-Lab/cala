from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Parameters(ABC):
    """Parameter management and validation"""

    def __post_init__(self) -> None:
        """Validate parameters after initialization"""
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate all parameters"""
        pass
