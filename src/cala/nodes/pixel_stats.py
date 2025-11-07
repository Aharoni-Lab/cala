import xarray as xr

from cala.assets import Frame, Movie, PixStats, PopSnap, Traces, Footprints
from cala.models import AXIS


def initialize(traces: Traces, frames: Movie) -> PixStats:
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

    pixel_stats_ = PixStats.from_array(W)
    return pixel_stats_


def ingest_frame(
    pixel_stats: PixStats, frame: Frame, new_traces: PopSnap, footprints: Footprints
) -> PixStats:
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
        new_traces (PopSnap): Current temporal component c_t.
            Shape: (components)
    """
    if new_traces.array is None:
        return pixel_stats

    # Compute scaling factors
    frame_idx = frame.array[AXIS.frame_coord].item()
    prev_scale = frame_idx / (frame_idx + 1)
    new_scale = 1 / (frame_idx + 1)

    y_t = frame.array
    W = pixel_stats.array
    c_t = new_traces.array  # New frame traces

    # A = footprints.array.transpose(AXIS.component_dim, ...)
    # coords = A.data.nonzero()
    # comps = coords[0]
    # segments = {}
    # for comp in np.unique(comps):
    #     segments[comp] = [
    #         coords[1][np.where(comps == comp)[0]],
    #         coords[2][np.where(comps == comp)[0]],
    #     ]
    # Update pixel-component statistics W_t
    # W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
    # We only access the footprint area, so we can drastically reduce the calc
    # pixel_stats should be sparse, too, I think.
    W_update = prev_scale * W + new_scale * y_t @ c_t

    pixel_stats.array = W_update.reset_coords([AXIS.timestamp_coord, AXIS.frame_coord], drop=True)

    return pixel_stats


def ingest_component(
    pixel_stats: PixStats, frames: Movie, new_traces: Traces, traces: Traces = None
) -> PixStats:
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
    c_new = new_traces.array
    if c_new is None:
        return pixel_stats

    W = pixel_stats.array
    if W is None:
        pixel_stats.array = initialize(traces, frames).array
        return pixel_stats

    # Compute scaling factor (1/t)
    frame_idx = c_new[AXIS.frame_coord].max().item()
    scale = 1 / (frame_idx + 1)

    # Compute outer product of frame and new traces
    # (1/t)Y_buf c_new^T
    new_stats = scale * (frames.array @ c_new)

    merged_ids = c_new.attrs.get("replaces")
    if merged_ids:
        intact_ids = [id_ for id_ in W[AXIS.id_coord].values if id_ not in merged_ids]
        W = W.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: intact_ids}).reset_index(AXIS.id_coord)

    pixel_stats.array = xr.concat([W, new_stats], dim=AXIS.component_dim)

    return pixel_stats
