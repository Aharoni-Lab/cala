from dataclasses import dataclass, field

import numpy as np


@dataclass
class Estimates:
    """Stores and manages all estimation results"""

    dimensions: tuple[int, ...]
    num_components: int = 0
    current_timestamp: int = 0

    shifts: list = field(default_factory=list)  # motion stabilization shifts

    spatial_footprints: np.ndarray = field(init=False)  # A
    temporal_traces: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # C

    background_footprints: np.ndarray = field(init=False)  # b
    background_traces: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # f

    spike_activity: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # S

    pixel_statistics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # CY
    source_statistics: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0))
    )  # CC

    def __post_init__(self):
        """Initialize default values for sparse matrices"""
        self.spatial_footprints = np.zeros(np.prod(self.dimensions))
        self.background_footprints = np.zeros(np.prod(self.dimensions))
