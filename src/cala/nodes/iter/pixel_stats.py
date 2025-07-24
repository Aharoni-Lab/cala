import xarray as xr

from cala.models import AXIS, Movie, Traces


class PixelStats:
    pixel_stats_: xr.DataArray = None
    """Updated pixel-component sufficient statistics W."""

    def initialize(self, traces: Traces, movie: Movie) -> xr.DataArray:
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
            movie (Movie): Stack of frames up to current timestep.
                Shape: (frames × height × width)
        """
        t_prime = movie.sizes[AXIS.frames_dim]

        Y = movie

        C = traces  # components x time

        # Compute W = Y[:, 1:t']C^T/t'
        W = Y @ C.T / t_prime

        self.pixel_stats_ = W
        return self.pixel_stats_

    def ingest_frame(self, frame: xr.DataArray, traces: Traces) -> xr.DataArray:
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
            traces (Traces): Current temporal component c_t.
                Shape: (components)
        """
        # Compute scaling factors
        frame_idx = frame.coords[AXIS.frame_coord].item()
        prev_scale = frame_idx / (frame_idx + 1)
        new_scale = 1 / (frame_idx + 1)

        y_t = frame
        W = self.pixel_stats_
        c_t = traces  # New frame traces

        # Update pixel-component statistics W_t
        # W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
        W_update = prev_scale * W + new_scale * y_t @ c_t

        self.pixel_stats_ = W_update

        return self.pixel_stats_

    def ingest_component(self, movie: Movie, new_traces: Traces = None) -> xr.DataArray:
        """Update pixel statistics with new components.

        Updates W_t according to the equation:

            W_t = [W_t, (1/t)Y_buf c_new^T]

        where:
            t is the current frame index.

        Args:
            movie (Movie): Stack of frames up to current timestep.
            new_traces (Traces): Newly detected components' traces

        Returns:
            PixelStats: Updated pixel statistics matrix
        """
        if new_traces is None:
            return self.pixel_stats_

        # Compute scaling factor (1/t)
        frame_idx = new_traces[AXIS.frame_coord][-1].item()
        scale = 1 / (frame_idx + 1)

        # Compute outer product of frame and new traces
        # (1/t)Y_buf c_new^T
        new_stats = scale * (movie @ new_traces)

        # Concatenate with existing pixel stats along component axis
        return xr.concat([self.pixel_stats_, new_stats], dim=AXIS.component_dim)
