from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FluorescentObject(ABC):
    """Base class for any fluorescent object detected in calcium imaging."""

    # Spatial and temporal properties
    footprint: np.ndarray
    """2D array representing spatial distribution"""
    time_trace: np.ndarray
    """1D array of fluorescence intensity over time"""

    # Basic metrics
    confidence_level: float
    """Confidence in object detection/separation"""
    snr: float
    """Signal to noise ratio"""
    spatial_correlation: float
    """Correlation within footprint"""

    # Temporal dynamics
    rise_time_constant: Optional[float] = None
    """Tau rise in seconds"""
    decay_time_constant: Optional[float] = None
    """Tau decay in seconds"""

    # Relationship with other objects
    overlapping_objects: List["FluorescentObject"] = field(default_factory=list)
    """Objects with overlapping footprints"""

    def __post_init__(self):
        self.overlapping_objects = (
            [] if self.overlapping_objects is None else self.overlapping_objects
        )

    def calculate_snr(self) -> float:
        """Calculate signal to noise ratio."""
        raise NotImplementedError

    def calculate_spatial_correlation(self) -> float:
        """Calculate correlation within footprint pixels."""
        raise NotImplementedError
