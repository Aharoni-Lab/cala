from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters, Traces
from cala.streaming.stores.odl import PixelStats


@dataclass
class PixelStatsInitializerParams(Parameters):
    """Parameters for pixel-component statistics computation.

    This class defines the configuration parameters needed for computing statistics
    across pixels and components, including axis names and spatial specifications.
    """

    component_axis: str = "components"
    """Name of the dimension representing individual components."""

    id_coordinates: str = "id_"
    """Name of the coordinate used to identify individual components with unique IDs."""

    type_coordinates: str = "type_"
    """Name of the coordinate used to specify component types (e.g., neuron, background)."""

    frames_axis: str = "frame"
    """Name of the dimension representing time points."""

    spatial_axes: tuple = ("height", "width")
    """Names of the dimensions representing spatial coordinates (height, width)."""

    def validate(self):
        """Validate parameter configurations.

        Raises:
            ValueError: If spatial_axes is not a tuple of length 2.
        """
        if not isinstance(self.spatial_axes, tuple) or len(self.spatial_axes) != 2:
            raise ValueError("spatial_axes must be a tuple of length 2")


@dataclass
class PixelStatsInitializer(SupervisedTransformer):
    """Computes pixel-component statistics using temporal components and frame data.

    This transformer calculates the correlation between each pixel's temporal trace
    and each component's temporal activity. The computation provides a measure of
    how well each pixel's activity aligns with each component.

    The computation follows the equation: W = Y[:, 1:t']C^T/t'
    where:
    - Y is the data matrix (pixels × time)
    - C is the temporal components matrix (components × time)
    - t' is the current timestep
    - W is the resulting pixel statistics (pixels × components)

    The result W represents the temporal correlation between each pixel
    and each component, normalized by the number of timepoints.
    """

    params: PixelStatsInitializerParams
    """Configuration parameters for the computation."""

    pixel_stats_: xr.DataArray = field(init=False)
    """Computed correlation between pixels and components."""

    def learn_one(self, traces: Traces, frame: xr.DataArray) -> Self:
        """Compute pixel-component statistics from frames and temporal components.

        This method implements the correlation computation between each pixel's
        temporal trace and each component's activity. The correlation is normalized
        by the current timestep to account for varying temporal lengths.

        Args:
            traces (Traces): Temporal traces of all detected fluorescent components.
                Shape: (components × time)
            frame (xr.DataArray): Stack of frames up to current timestep.
                Shape: (frames × height × width)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Get current timestep
        t_prime = frame.sizes[self.params.frames_axis]

        # Reshape frames to pixels x time
        Y = frame.values.reshape(-1, t_prime)

        # Get temporal components C
        C = traces.values  # components x time

        # Compute W = Y[:, 1:t']C^T/t'
        W = Y @ C.T / t_prime

        # Reshape W back to spatial dimensions x components
        W = W.reshape(*[frame.sizes[ax] for ax in self.params.spatial_axes], -1)

        # Create xarray DataArray with proper dimensions and coordinates
        self.pixel_stats_ = xr.DataArray(
            W,
            dims=(*self.params.spatial_axes, self.params.component_axis),
            coords={
                self.params.id_coordinates: (
                    self.params.component_axis,
                    traces.coords[self.params.id_coordinates].values,
                ),
                self.params.type_coordinates: (
                    self.params.component_axis,
                    traces.coords[self.params.type_coordinates].values,
                ),
            },
        )

        return self

    def transform_one(self, _=None) -> PixelStats:
        """Transform the computed statistics into the expected format.

        This method reshapes the pixel statistics to match the expected
        dimensions order (components × height × width) and wraps them
        in a PixelStats object for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            PixelStats: Wrapped pixel-wise statistics with proper dimensionality.
        """
        return self.pixel_stats_.transpose(
            self.params.component_axis, *self.params.spatial_axes
        )
