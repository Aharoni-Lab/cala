from dataclasses import dataclass

from .base import FluorescentObject
from .neuron import Neuron


@dataclass
class Background(FluorescentObject):
    """Class representing background components in calcium imaging."""

    background_type: str = "neuropil"
    """Type of background (neuropil, blood vessel, etc.)"""

    def estimate_contamination(self, neuron: Neuron) -> float:
        """Estimate contamination of this background component on a neuron."""
        raise NotImplementedError
