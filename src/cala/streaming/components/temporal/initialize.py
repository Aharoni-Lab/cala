from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from numba import jit, prange
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
        # Get frames to use and flatten them
        flattened_frames = frames[: self.params.num_frames_to_use].values.reshape(
            self.params.num_frames_to_use, -1
        )

        # Process all components at once using Numba parallel
        self.temporal_traces_ = solve_all_component_traces(
            estimates.spatial_footprints, flattened_frames
        )

        return self

    def transform_one(self, estimates: Estimates) -> Estimates:
        """Transform method assigns to estimates."""
        estimates.temporal_traces = self.temporal_traces_
        return estimates


@jit(nopython=True, cache=True, parallel=True)
def solve_all_component_traces(footprints, frames):
    """Solve temporal traces for all components in parallel

    Args:
        footprints: Array of shape (n_components, height*width)
        frames: Array of shape (n_frames, height*width)
    Returns:
        Array of shape (n_components, n_frames)
    """
    n_components = footprints.shape[0]
    n_frames = frames.shape[0]
    results = np.zeros((n_components, n_frames))

    # Parallel loop over components
    for i in prange(n_components):
        footprint = footprints[i].reshape(-1)
        active_pixels = footprint > 0

        if np.any(active_pixels):
            footprint_active = footprint[active_pixels]
            frames_active = frames[:, active_pixels]
            results[i] = fast_nnls_vector(footprint_active, frames_active)

    return results


@jit(nopython=True, cache=True, fastmath=True)
def fast_nnls_vector(A, B):
    """Specialized NNLS for single-variable case across multiple frames
    A: footprint values (n_pixels,)
    B: frame data matrix (n_frames, n_pixels)
    Returns: brightness values for each frame (n_frames,)
    """
    ata = (A * A).sum()  # Compute once for all frames
    if ata <= 0:
        return np.zeros(B.shape[0])

    # Vectorized computation for all frames
    atb = A @ B.T  # dot product with each frame
    return np.maximum(0, atb / ata)
