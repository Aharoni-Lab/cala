from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from numba import jit, prange
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters
from cala.streaming.initialization.meta import TransformerMeta
from cala.streaming.types import Footprints, Traces


@dataclass
class TracesInitializerParams(Parameters):
    """Parameters for traces initialization"""

    component_axis: str = "components"
    """Axis for components"""
    frames_axis: str = "frames"
    """Axis for frames"""

    num_frames_to_use: int = 3
    """Number of frames to use for temporal initialization"""

    def validate(self):
        if not self.num_frames_to_use > 0:
            raise ValueError("Parameter num_frames_to_use must be a positive integer.")


@dataclass
class TracesInitializer(SupervisedTransformer, metaclass=TransformerMeta):
    """Initializes temporal components using projection methods."""

    params: TracesInitializerParams
    """Parameters for temporal initialization"""

    def learn_one(
        self,
        footprints: Footprints,
        frames: xr.DataArray,
    ) -> Self:
        """Learn temporal traces from footprints and frames."""
        # Get frames to use and flatten them
        n_frames = min(
            frames.sizes[self.params.frames_axis], self.params.num_frames_to_use
        )
        flattened_frames = frames[:n_frames].values.reshape(n_frames, -1)

        # Process all components
        temporal_traces = solve_all_component_traces(
            footprints.values.reshape(footprints.sizes[self.params.component_axis], -1),
            flattened_frames,
        )

        # Store result
        self.traces_ = xr.DataArray(
            temporal_traces,
            dims=(self.params.component_axis, self.params.frames_axis),
            coords={
                self.params.component_axis: footprints.coords[
                    self.params.component_axis
                ],
                self.params.frames_axis: frames.coords[self.params.frames_axis][
                    :n_frames
                ],
            },
        )
        return self

    def transform_one(self, footprints: xr.DataArray) -> Traces:
        """Return initialization result."""
        return Traces(self.traces_)


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
