from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .base import FluorescentObject


@dataclass
class Neuron(FluorescentObject):
    """Class representing a detected neuron in calcium imaging."""

    deconvolved_signal: Optional[np.ndarray] = None
    """Deconvolved neural activity"""
    spike_times: Optional[np.ndarray] = None
    """Estimated spike times"""
    cell_type: Optional[str] = None
    """Identified cell type if available"""
    metadata: Optional[Dict] = field(default_factory=dict)
    """Additional metadata"""
    rise_time_constant: Optional[float] = None
    """Tau rise in seconds"""
    decay_time_constant: Optional[float] = None
    """Tau decay in seconds"""

    def classify_cell_type(self) -> str:
        """Classify neuron type based on its properties."""
        raise NotImplementedError
