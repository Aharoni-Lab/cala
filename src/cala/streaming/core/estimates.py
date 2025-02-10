from dataclasses import dataclass, field

import numpy as np


@dataclass
class Estimates:
    """Stores and manages all estimation results"""

    spatial_components: np.ndarray  # A
    temporal_components: np.ndarray  # C
    background_spatial: np.ndarray  # b
    background_temporal: np.ndarray  # f
    neural_activity: np.ndarray  # S
    noise_levels: np.ndarray  # sn
    pixel_statistics: np.ndarray  # CY
    source_statistics: np.ndarray  # CC
    shifts: list = field(default_factory=list)  # motion correction shifts
