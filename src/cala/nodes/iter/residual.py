import xarray as xr
from noob.node import Node

from cala.models import AXIS, Footprints, Frame, Movie, PopSnap, Residual, Traces


class Resident(Node):
    """
    Computes and maintains a buffer of residual signals.

    This method implements the residual computation by subtracting the
    reconstructed signal from the original data. It maintains only the
    most recent frames as specified by the buffer length.

    The residual buffer contains the recent history of unexplained variance
    in the data after accounting for known components.
    """

    residual_: Residual = None
    """Computed residual buffer containing recent unexplained signals."""

    def process(
        self,
        footprints: Footprints,
        traces: Traces | PopSnap,
        frames: Movie = None,
        frame: Frame = None,
    ) -> Residual:
        if frame is None:
            return self.initialize(footprints=footprints, traces=traces, frames=frames)
        else:
            return self.ingest_frame(footprints=footprints, traces=traces, frame=frame)

    def initialize(self, frames: Movie, footprints: Footprints, traces: Traces) -> Residual:
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
            frames (Movie): Stack of frames up to current timestep.
                Shape: (frames × height × width)
        """
        # Reshape frames to pixels x time
        Y = frames.array

        # Get temporal components [C; f]
        C = traces.array  # components x time

        # Reshape footprints to (pixels x components)
        A = footprints.array

        # Compute residual R = Y - [A,b][C;f]
        R = Y - (A @ C)

        self.residual_ = Residual(array=R)
        return self.residual_

    def ingest_frame(self, frame: Frame, footprints: Footprints, traces: PopSnap) -> Residual:
        """
        Update residual buffer with new frame, footprints, and traces
        """
        new_R = frame.array - footprints.array @ traces.array

        self.residual_.array = xr.concat([self.residual_.array, new_R], dim=AXIS.frames_dim)

        return self.residual_
