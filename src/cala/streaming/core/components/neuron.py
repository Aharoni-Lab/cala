from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .base import FluorescentObject


@dataclass
class Neuron(FluorescentObject):
    """Class representing a detected neuron in calcium imaging."""

    # Neuron-specific properties
    deconvolved_signal: Optional[np.ndarray] = None  # deconvolved neural activity
    spike_times: Optional[np.ndarray] = None  # estimated spike times
    cell_type: Optional[str] = None  # identified cell type if available
    metadata: Dict = field(default_factory=dict)  # additional metadata

    def deconvolve_signal(self, method: str = "default") -> np.ndarray:
        """Deconvolve calcium signal to estimate neural activity."""
        raise NotImplementedError

    def detect_spikes(self, threshold: float = 2.0) -> np.ndarray:
        """Detect spike times from deconvolved signal."""
        raise NotImplementedError

    def classify_cell_type(self) -> str:
        """Classify neuron type based on its properties."""
        raise NotImplementedError
