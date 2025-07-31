import xarray as xr
from noob.node import Node

from cala.assets import Frame, Movie, PixStats, PopSnap, Traces
from cala.models import AXIS


class PixelStater(Node):
    pixel_stats_: PixStats = None
    """Updated pixel-component sufficient statistics W."""

    def process(
        self,
        traces: Traces | PopSnap = None,
        frames: Movie = None,
        new_traces: Traces = None,
        frame: Frame = None,
    ) -> PixStats:
        if frames is None:
            return self.ingest_frame(frame=frame, traces=traces)
        elif new_traces is None:
            return self.initialize(frames=frames, traces=traces)
        else:
            return self.ingest_components(frames=frames, new_traces=new_traces)

    def initialize(self, traces: Traces, frames: Movie) -> PixStats:
        """
        This transformer calculates the correlation between each pixel's temporal trace
        and each component's temporal activity. The computation provides a measure of
        how well each pixel's activity aligns with each component.

        The computation follows the equation:

            W = Y[:, 1:t']C^T/t'

            where:
            - Y is the data matrix (pixels × time)
            - C is the temporal components matrix (components × time)
            - t' is the current timestep
            - W is the resulting pixel statistics (pixels × components)

        The result W represents the temporal correlation between each pixel
        and each component, normalized by the number of timepoints.

        Args:
            traces (Traces): Temporal traces of all detected fluorescent components.
                Shape: (components × time)
            frames (Movie): Stack of frames up to current timestep.
                Shape: (frames × height × width)
        """
        Y = frames.array

        t_prime = Y[AXIS.frame_coord].max().item() + 1

        C = traces.array  # components x time

        # Compute W = Y[:, 1:t']C^T/t'
        W = Y @ C.T / t_prime

        self.pixel_stats_ = PixStats(array=W)
        return self.pixel_stats_

    def ingest_frame(self, frame: Frame, traces: PopSnap) -> PixStats:
        """
        Update pixel statistics using current frame and component.

        This method implements the update equations for pixel-component wise
        statistics matrices. The updates incorporate the current frame and
        temporal component with appropriate time-based scaling.

            W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T

            where:
            - W_t is the pixel-wise sufficient statistics at time t
            - y_t is the current frame
            - c_t is the current temporal component
            - t is the current timestep

        Args:
            frame (Frame): Current frame y_t.
                Shape: (height × width)
            traces (PopSnap): Current temporal component c_t.
                Shape: (components)
        """
        # Compute scaling factors
        frame_idx = frame.array.coords[AXIS.frame_coord].item()
        prev_scale = frame_idx / (frame_idx + 1)
        new_scale = 1 / (frame_idx + 1)

        y_t = frame.array
        W = self.pixel_stats_.array
        c_t = traces.array  # New frame traces

        # Update pixel-component statistics W_t
        # W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
        W_update = prev_scale * W + new_scale * y_t @ c_t

        self.pixel_stats_.array = W_update.reset_coords(
            [AXIS.timestamp_coord, AXIS.frame_coord], drop=True
        )

        return self.pixel_stats_

    def ingest_component(self, frames: Movie, new_traces: Traces) -> PixStats:
        """Update pixel statistics with new components.

        Updates W_t according to the equation:

            W_t = [W_t, (1/t)Y_buf c_new^T]

        where:
            t is the current frame index.

        Args:
            frames (Movie): Stack of frames up to current timestep.
            new_traces (Traces): Newly detected components' traces

        Returns:
            PixelStater: Updated pixel statistics matrix
        """
        # Compute scaling factor (1/t)
        frame_idx = new_traces.array[AXIS.frame_coord].max().item()
        scale = 1 / (frame_idx + 1)

        # Compute outer product of frame and new traces
        # (1/t)Y_buf c_new^T
        new_stats = scale * (frames.array @ new_traces.array)

        # Concatenate with existing pixel stats along component axis
        self.pixel_stats_.array = xr.concat(
            [self.pixel_stats_.array, new_stats], dim=AXIS.component_dim
        )

        return self.pixel_stats_
