from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Traces
from cala.streaming.stores.odl import ComponentStats


@dataclass
class ComponentStatsInitializerParams(Parameters, Axis):
    """Parameters for component statistics computation.

    This class defines the configuration parameters needed for computing statistics
    across components, including axis names and coordinate specifications.
    """

    def validate(self) -> None:
        """Validate parameter configurations.

        Raises:
            ValueError: If spatial_axes is not a tuple of length 2.
        """
        pass


@dataclass
class ComponentStatsInitializer(SupervisedTransformer):
    """Computes correlation statistics between temporal components.

    This transformer calculates the correlation matrix between temporal components
    using their activity traces. The correlation is computed as a normalized
    outer product of the temporal components.

    The computation follows the equation:  M = C @ C.T / t'
    where:
    - C is the temporal components matrix (components × time)
    - t' is the current timestep
    - M is the resulting correlation matrix (components × components)
    """

    params: ComponentStatsInitializerParams
    """Configuration parameters for the computation."""

    component_stats_: xr.DataArray = field(init=False)
    """Computed correlation matrix between components."""

    def learn_one(self, traces: Traces, frame: xr.DataArray) -> Self:
        """Compute correlation statistics from temporal components.

        This method implements the correlation computation between components
        using their temporal traces. The correlation matrix is normalized by
        the current timestep.

        Args:
            traces (Traces): Temporal traces of all detected fluorescent components.
                Shape: (components × time)
            frame (xr.DataArray): Current frame data, used for temporal normalization.
                Shape: (frames × height × width)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Get current timestep
        t_prime = frame.sizes[self.params.frames_axis]

        # Get temporal components C
        C = traces  # components x time

        # Compute M = C * C.T / t'
        M = C @ C.rename({self.params.component_axis: f"{self.params.component_axis}'"}) / t_prime

        # Create xarray DataArray with proper dimensions and coordinates
        self.component_stats_ = M.assign_coords(C.coords[Axis.component_axis].coords)

        return self

    def transform_one(self, _: None = None) -> ComponentStats:
        """Return the computed component statistics.

        This method wraps the computed correlation matrix in a ComponentStats
        object for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            ComponentStats: Wrapped correlation matrix between components.
        """
        return self.component_stats_
