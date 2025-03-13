from dataclasses import dataclass, field
from typing import Self

import numpy as np
import xarray as xr
from numba import jit, prange
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

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

    def validate(self):
        pass


@dataclass
class TracesInitializer(SupervisedTransformer, metaclass=TransformerMeta):
    """Initializes temporal components using projection methods."""

    params: TracesInitializerParams
    """Parameters for temporal initialization"""
    traces_: Traces = field(init=False, repr=False)

    is_fitted_: bool = False

    def learn_one(
        self,
        footprints: Footprints,
        frame: xr.DataArray,
    ) -> Self:
        """Learn temporal traces from footprints and frames."""
        if footprints.isel({self.params.component_axis: 0}).shape != frame[0].shape:
            raise ValueError("Footprint and frame dimensions must be identical.")

        # Get frames to use and flatten them
        n_frames = frame.sizes[self.params.frames_axis]
        flattened_frames = frame[:n_frames].values.reshape(n_frames, -1)
        flattened_footprints = footprints.values.reshape(
            footprints.sizes[self.params.component_axis], -1
        )

        # Process all components
        temporal_traces = solve_all_component_traces(
            flattened_footprints,
            flattened_frames,
        )

        # Store result
        self.traces_ = Traces(
            temporal_traces,
            dims=(self.params.component_axis, self.params.frames_axis),
            coords={
                self.params.component_axis: footprints.coords[
                    self.params.component_axis
                ],
                self.params.frames_axis: frame.coords[self.params.frames_axis][
                    :n_frames
                ],
            },
        )

        self.is_fitted_ = True
        return self

    def transform_one(self, _=None) -> Traces:
        """Return initialization result."""
        if not self.is_fitted_:
            raise NotFittedError

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
