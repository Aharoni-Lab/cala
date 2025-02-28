from enum import Enum, auto


class ComponentType(Enum):
    """Enum representing different types of fluorescent components."""

    NEURON = auto()
    BACKGROUND = auto()
