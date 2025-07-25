import xarray as xr
from noob.node import Node

from cala.models import AXIS, Footprints, Frame, Traces


class Residuals(Node):
    """
    Computes and maintains a buffer of residual signals.

    This method implements the residual computation by subtracting the
    reconstructed signal from the original data. It maintains only the
    most recent frames as specified by the buffer length.

    The residual buffer contains the recent history of unexplained variance
    in the data after accounting for known components.
    """

    buffer_length: int

    residual_: xr.DataArray = None
    """Computed residual buffer containing recent unexplained signals."""

    def initialize(
        self, footprints: Footprints, traces: Traces, frame: xr.DataArray
    ) -> xr.DataArray:
        """
        The computation follows the equation:
            R_buf = [Y − [A, b][C; f]][:, t′ − l_b + 1 : t′]
            where:
            - Y is the data matrix (pixels × time)
            - [A, b] is the spatial footprint matrix of neurons and background
            - [C; f] is the temporal traces matrix of neurons and background
            - t' is the current timestep
            - l_b is the buffer length
            - R_buf is the resulting residual buffer

        Args:
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)
            traces (Traces): Temporal traces of all components.
                Shape: (components × time)
            frame (xr.DataArray): Stack of frames up to current timestep.
                Shape: (frames × height × width)
        """
        # Get current timestep
        t_prime = frame.sizes[AXIS.frames_dim]

        # Reshape frames to pixels x time
        Y = frame.stack({"pixels": AXIS.spatial_dims})

        # Get temporal components [C; f]
        C = traces  # components x time

        # Reshape footprints to (pixels x components)
        A = footprints.stack({"pixels": AXIS.spatial_dims})

        # Compute residual R = Y - [A,b][C;f]
        R = Y - (A @ C)

        # Only keep the last l_b frames
        start_idx = max(0, t_prime - self.params.buffer_length)
        R = R.isel({AXIS.frames_dim: slice(start_idx, None)})

        self.residual_ = R.unstack("pixels")
        return self.residual_

    def update(self, frame: Frame, footprints: Footprints, traces: Traces) -> xr.DataArray:
        """
        Update residual buffer with new frame, footprints, and traces..

        :param frame:
        :param footprints:
        :param traces:
        :return:
        """
        new_R = frame.array - footprints.array @ traces.array.sel(
            {AXIS.frames_coord: frame.array[AXIS.frames_coord]}
        )

        self.residual_ = xr.concat([self.residual_, new_R], dim=AXIS.frames_dim).isel(
            {AXIS.frames_dim: slice(-self.buffer_length, None)}
        )

        return self.residual_
