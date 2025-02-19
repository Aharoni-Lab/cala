from typing import List, Optional, Self

import numpy as np
from river import compose

from ..components.deconvolution import OASIS
from ..components.spatial import SpatialComponentUpdater
from ..components.temporal import TemporalComponentUpdater
from ..core.estimates import Estimates
from ..core.parameters import Parameters
from ..detection.new_components import NewComponentDetector
from ..motion_stabilization import RigidTranslator


class StreamingCNMF:
    """Main interface class that orchestrates the streaming CNMF processing"""

    def __init__(self, params: Parameters, estimates: Optional[Estimates] = None):
        """Initialize StreamingCNMF algorithm

        Args:
            params: Parameters for the algorithm
            estimates: Optional pre-existing estimates
        """
        self.params = params
        self.estimates = estimates if estimates is not None else Estimates()
        self.timestep = 0  # Current timestep

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all processing components"""
        self.motion_stabilizer = RigidTranslator(self.params.motion_params)
        self.spatial_updater = SpatialComponentUpdater()
        self.temporal_updater = TemporalComponentUpdater()
        self.deconvolver = OASIS()
        self.component_detector = NewComponentDetector(self.params.detection_params)

        self.preprocess = compose.Pipeline(self.motion_stabilizer)
        self.demix = compose.Pipeline(
            self.spatial_updater,
            self.temporal_updater,
            self.component_detector,
            self.deconvolver,
        )

    def process_frame(self, frame: np.ndarray) -> Estimates:
        """Process a single frame

        Args:
            frame: Input frame to process

        Returns:
            Updated estimates
        """
        self.preprocess.learn_one(frame)
        frame, shift = self.preprocess.transform_one(frame)
        self.estimates.shifts.append(shift)

        self.demix.learn_one(frame, self.estimates)
        self.estimates = self.demix.transform_one(frame, self.estimates)

        self.timestep += 1
        return self.estimates

    def fit(self, movie_files: List[str]) -> Self:
        """Process all frames from provided movie files

        Args:
            movie_files: List of paths to movie files

        Returns:
            self for method chaining
        """
        for file in movie_files:
            movie = np.load(file)  # use proper movie loading here
            for frame in movie:
                self.process_frame(frame)
        return self
