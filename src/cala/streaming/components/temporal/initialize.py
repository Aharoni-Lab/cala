from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from scipy.optimize import nnls

from cala.streaming.core import Estimates, Parameters


@dataclass
class TemporalInitializerParams(Parameters):
    """Parameters for temporal initialization"""

    num_frames_to_use: int = 3

    def validate(self):
        if not self.num_frames_to_use > 0:
            raise ValueError("Parameter num_frames_to_use must be a positive integer.")


class TemporalInitializer(SupervisedTransformer):
    """Initializes temporal components using projection methods.

    For each spatial footprint, finds temporal traces by minimizing the reconstruction error
    within each footprint's active area.
    """

    def __init__(self, params: TemporalInitializerParams):
        self.params = params
        self.temporal_traces_ = None

    def learn_one(self, estimates: Estimates, frames: xr.DataArray) -> Self:
        """Learn temporal traces from a batch of frames using least squares optimization.

        For each component, finds the temporal trace values that minimize the reconstruction error:
        min ||Y - a*c||^2 where:
        - Y is the frame data within the footprint's active area
        - a is the footprint values in the active area
        - c is the temporal trace value to solve for

        Args:
            estimates: Estimates object containing spatial footprints and other parameters
            frames: xarray DataArray of shape (frames, height, width) containing 2D grayscale frames

        Returns:
            self
        """
        # Get number of components
        num_components = estimates.spatial_footprints.shape[0]

        # Get frames to use and flatten them
        flattened_frames = frames[: self.params.num_frames_to_use].values.reshape(
            self.params.num_frames_to_use, -1
        )

        # Initialize temporal traces matrix
        self.temporal_traces_ = np.zeros(
            (num_components, self.params.num_frames_to_use)
        )

        # For each component, solve least squares problem
        for comp_idx, footprint in enumerate(estimates.spatial_footprints):
            # Reshape footprint to match frame dimensions
            footprint = footprint.reshape(-1)

            # Get active pixels in footprint (where footprint > 0)
            active_pixels = footprint > 0

            if np.any(active_pixels):
                # Extract active areas from footprint and frames
                footprint_active = footprint[active_pixels]
                frames_active = flattened_frames[:, active_pixels]

                # For each frame, solve least squares to find optimal temporal trace value
                for frame_idx, frame_active in enumerate(frames_active):
                    # Solve non-negative least squares: min ||y - a*c||^2 subject to c >= 0
                    # where y is frame data, a is footprint values, c is temporal trace value
                    temporal_trace, _ = nnls(
                        footprint_active.reshape(-1, 1), frame_active
                    )
                    self.temporal_traces_[comp_idx, frame_idx] = temporal_trace[0]

        return self

    def transform_one(self, estimates: Estimates) -> Estimates:
        """Transform method assigns to estimates."""
        estimates.temporal_traces = self.temporal_traces_
        return estimates
