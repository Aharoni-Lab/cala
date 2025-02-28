from dataclasses import dataclass, field
from typing import Self

import numpy as np
import xarray as xr
from numba import jit, prange
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters
from cala.streaming.core.components import ComponentManager


@dataclass
class TemporalInitializerParams(Parameters):
    """Parameters for temporal initialization"""

    component_axis: str = "component"
    """Axis for components"""
    frames_axis: str = "frames"
    """Spatial axes for footprints"""

    num_frames_to_use: int = 3
    """Number of frames to use for temporal initialization"""

    def validate(self):
        if not self.num_frames_to_use > 0:
            raise ValueError("Parameter num_frames_to_use must be a positive integer.")


@dataclass
class TemporalInitializer(SupervisedTransformer):
    """Initializes temporal components using projection methods.

    For each spatial footprint, finds temporal traces by minimizing the reconstruction error
    within each footprint's active area.
    """

    params: TemporalInitializerParams
    """Parameters for temporal initialization"""
    temporal_traces_: xr.DataArray = field(init=False)
    """Temporal traces"""

    def learn_one(self, components: ComponentManager, frames: xr.DataArray) -> Self:
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
        temporal_traces = solve_all_component_traces(
            components.footprints.values, flattened_frames
        )

        self.temporal_traces_ = xr.DataArray(
            temporal_traces,
            dims=(self.params.component_axis, self.params.frames_axis),
            coords={
                self.params.component_axis: list(components.component_ids),
                self.params.frames_axis: frames.coords[self.params.frames_axis],
            },
        )

        return self

    def transform_one(self, components: ComponentManager) -> ComponentManager:
        """Transform method assigns to estimates."""

        components.populate_from_traces(self.temporal_traces_)
        return components


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
