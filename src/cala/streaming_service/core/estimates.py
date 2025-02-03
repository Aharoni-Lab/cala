class Estimates:
    """Stores and manages all estimation results"""

    def __init__(self):
        self.spatial_components = None  # A
        self.temporal_components = None  # C
        self.background_spatial = None  # b
        self.background_temporal = None  # f
        self.neural_activity = None  # S
        self.noise_levels = None  # sn
        self.shifts = []  # motion correction shifts
        self.pixel_statistics = None  # CY
        self.source_statistics = None  # CC
