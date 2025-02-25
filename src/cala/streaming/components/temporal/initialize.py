from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer

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

    For each spatial footprint, calculates the mean pixel values within the footprint
    across all input frames to initialize the temporal traces.
    """

    def __init__(self, params: TemporalInitializerParams):
        self.params = params
        self.temporal_traces_ = None

    def learn_one(self, estimates: Estimates, frames: xr.DataArray) -> Self:
        """Learn temporal traces from a batch of frames.

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

        # Ensure footprints are properly shaped
        footprints = estimates.spatial_footprints.reshape(num_components, -1)

        # Calculate normalization factors (sum of positive weights for each footprint)
        normalization = np.sum(footprints > 0, axis=1)
        normalization = np.where(
            normalization > 0, normalization, 1
        )  # Avoid division by zero

        # Calculate weighted sum for all components and frames at once
        # frames shape: (num_frames, pixels)
        # footprints shape: (num_components, pixels)
        # Result shape: (num_components, num_frames)
        self.temporal_traces_ = (footprints @ flattened_frames.T) / normalization[
            :, None
        ]

        return self

    def transform_one(self, estimates: Estimates) -> Estimates:
        """Transform method assigns to estimates."""
        estimates.temporal_traces = self.temporal_traces_
        return estimates
