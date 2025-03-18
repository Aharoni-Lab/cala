from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters, Traces, TransformerMeta
from cala.streaming.stores.odl import ComponentStats


@dataclass
class ComponentStatsInitializerParams(Parameters):
    """Parameters for component statistics computation"""

    component_axis: str = "components"
    """Axis for components"""
    id_coordinates: str = "id_"
    type_coordinates: str = "type_"
    frames_axis: str = "frame"
    """Frames axis"""
    spatial_axes: tuple = ("height", "width")
    """Spatial axes for pixel statistics"""

    def validate(self):
        if not isinstance(self.spatial_axes, tuple) or len(self.spatial_axes) != 2:
            raise ValueError("spatial_axes must be a tuple of length 2")


@dataclass
class ComponentStatsInitializer(SupervisedTransformer, metaclass=TransformerMeta):
    """Computes pixel statistics using temporal components.

    Implements the equation:  M = C * C.T / t'
    where:
    - C is the temporal components matrix
    - t' is the current timestep
    """

    params: ComponentStatsInitializerParams
    """Parameters for component statistics computation"""
    component_stats_: xr.DataArray = field(init=False)
    """Computed component statistics"""

    def learn_one(self, traces: Traces, frame: xr.DataArray) -> Self:
        """Learn pixel statistics from frames and temporal components.

        Args:
            traces: traces of all detected fluorescent components
            frame: xarray DataArray of shape (frames, height, width) containing 2D frames

        Returns:
            self
        """
        # Get current timestep
        t_prime = frame.sizes[self.params.frames_axis]

        # Get temporal components C
        C = traces.values  # components x time

        # Compute M = C * C.T / t'
        M = C @ C.T / t_prime

        # Reshape M back to components x components
        M = M.reshape(traces.sizes[self.params.component_axis], -1)

        # Create xarray DataArray with proper dimensions and coordinates
        self.component_stats_ = xr.DataArray(
            M,
            dims=(self.params.component_axis, self.params.component_axis),
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

    def transform_one(self, _=None) -> ComponentStats:
        """Transform method updates component footprints with computed statistics.

        Args:

        Returns:
            New ComponentStats
        """
        # Transpose to match expected footprint dimensions (components, height, width)
        return ComponentStats(self.component_stats_)
