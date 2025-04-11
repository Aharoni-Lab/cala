from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints, Traces
from cala.streaming.stores.odl import Residuals


@dataclass
class ResidualInitializerParams(Parameters, Axis):
    """Parameters for residual signal computation.

    This class defines the configuration parameters needed for computing and maintaining
    a buffer of residual signals, including axis names, spatial specifications, and
    buffer characteristics.
    """

    buffer_length: int = 50
    """Number of recent frames to maintain in the residual buffer (l_b)."""

    def validate(self) -> None:
        """Validate parameter configurations.

        Raises:
            ValueError: If buffer_length is not positive.
        """
        if self.buffer_length <= 0:
            raise ValueError("buffer_length must be positive")


@dataclass
class ResidualInitializer(SupervisedTransformer):
    """Computes and maintains a buffer of residual signals.

    This transformer calculates the residual signal by subtracting the reconstructed
    signal (using components and their temporal activities) from the original data.
    It maintains a buffer of recent residual frames for ongoing analysis.

    The computation follows the equation: R_buf = [Y − [A, b][C; f]][:, t′ − l_b + 1 : t′]
    where:
    - Y is the data matrix (pixels × time)
    - [A, b] is the spatial footprint matrix of neurons and background
    - [C; f] is the temporal traces matrix of neurons and background
    - t' is the current timestep
    - l_b is the buffer length
    - R_buf is the resulting residual buffer

    The residual buffer contains the recent history of unexplained variance
    in the data after accounting for known components.
    """

    params: ResidualInitializerParams
    """Configuration parameters for the residual computation."""

    residual_: xr.DataArray = field(init=False)
    """Computed residual buffer containing recent unexplained signals."""

    def learn_one(self, footprints: Footprints, traces: Traces, frame: xr.DataArray) -> Self:
        """Compute residual signals from frames, components, and their activities.

        This method implements the residual computation by subtracting the
        reconstructed signal from the original data. It maintains only the
        most recent frames as specified by the buffer length.

        Args:
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)
            traces (Traces): Temporal traces of all components.
                Shape: (components × time)
            frame (xr.DataArray): Stack of frames up to current timestep.
                Shape: (frames × height × width)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Get current timestep
        t_prime = frame.sizes[self.params.frames_axis]

        # Reshape frames to pixels x time
        Y = frame.stack({"pixels": self.params.spatial_axes})

        # Get temporal components [C; f]
        C = traces  # components x time

        # Reshape footprints to (pixels x components)
        A = footprints.stack({"pixels": self.params.spatial_axes})

        # Compute residual R = Y - [A,b][C;f]
        R = Y - (A @ C)

        # Only keep the last l_b frames
        start_idx = max(0, t_prime - self.params.buffer_length)
        R = R.isel({self.params.frames_axis: slice(start_idx, None)})

        self.residual_ = R.unstack("pixels")
        return self

    def transform_one(self, _: None = None) -> Residuals:
        """Return the computed residual buffer.

        This method wraps the residual buffer in a Residual object
        for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            Residuals: Wrapped residual buffer containing recent unexplained signals.
        """
        return self.residual_
