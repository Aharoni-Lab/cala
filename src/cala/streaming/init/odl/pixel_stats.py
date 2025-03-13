from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters
from cala.streaming.types import Traces
from cala.streaming.types.odl import PixelStats


@dataclass
class PixelStatsParams(Parameters):
    """Parameters for pixel statistics computation"""

    component_axis: str = "components"
    """Axis for components"""
    frames_axis: str = "frames"
    """Frames axis"""
    spatial_axes: tuple = ("height", "width")
    """Spatial axes for pixel statistics"""

    def validate(self):
        if not isinstance(self.spatial_axes, tuple) or len(self.spatial_axes) != 2:
            raise ValueError("spatial_axes must be a tuple of length 2")


@dataclass
class PixelStatsTransformer(SupervisedTransformer):
    """Computes pixel statistics using temporal components.

    Implements the equation: W = Y[:, 1:t']C^T/t'
    where:
    - Y is the data matrix (pixels x time)
    - C is the temporal components matrix
    - t' is the current timestep
    - W is the resulting pixel statistics
    """

    params: PixelStatsParams
    """Parameters for pixel statistics computation"""
    pixel_stats_: xr.DataArray = field(init=False)
    """Computed pixel statistics"""

    def learn_one(self, traces: Traces, frames: xr.DataArray) -> Self:
        """Learn pixel statistics from frames and temporal components.

        Args:
            traces: traces of all detected fluorescent components
            frames: xarray DataArray of shape (frames, height, width) containing 2D frames

        Returns:
            self
        """
        # Get current timestep
        t_prime = frames.sizes[self.params.frames_axis]

        # Reshape frames to pixels x time
        Y = frames.values.reshape(-1, t_prime)

        # Get temporal components C
        C = traces.values  # components x time

        # Compute W = Y[:, 1:t']C^T/t'
        W = Y @ C.T / t_prime

        # Reshape W back to spatial dimensions x components
        W = W.reshape(*[frames.sizes[ax] for ax in self.params.spatial_axes], -1)

        # Create xarray DataArray with proper dimensions and coordinates
        self.pixel_stats_ = xr.DataArray(
            W,
            dims=(*self.params.spatial_axes, self.params.component_axis),
            coords={
                **{ax: frames.coords[ax] for ax in self.params.spatial_axes},
                self.params.component_axis: traces.coords[self.params.component_axis],
            },
        )

        return self

    def transform_one(self, _=None) -> PixelStats:
        """Transform method updates component footprints with computed statistics.

        Args:

        Returns:
            Updated ComponentManager
        """
        # Transpose to match expected footprint dimensions (components, height, width)
        return PixelStats(
            self.pixel_stats_.transpose(
                self.params.component_axis, *self.params.spatial_axes
            )
        )
