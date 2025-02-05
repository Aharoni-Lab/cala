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


@dataclass
class DataParameters(Parameters):
    """Data parameters"""

    def _validate_parameters(self) -> None:
        """Validate data parameters"""
        pass


@dataclass
class StreamingParameters(Parameters):
    """Streaming parameters"""

    def _validate_parameters(self) -> None:
        """Validate streaming parameters"""
        pass


@dataclass
class MotionParameters(Parameters):
    """Motion parameters"""

    def _validate_parameters(self) -> None:
        """Validate motion parameters"""
        pass


@dataclass
class InitializationParameters(Parameters):
    """Initialization parameters"""

    def _validate_parameters(self) -> None:
        """Validate initialization parameters"""
        pass
