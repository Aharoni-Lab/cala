import numpy as np
import xarray as xr

from cala.assets import CompStats, Frame, PopSnap, Trace, Traces
from cala.models import AXIS


def initialize(traces: Traces) -> CompStats:
    """
    calculates the correlation matrix between temporal components
    using their activity traces. The correlation is computed as a normalized
    outer product of the temporal components.

        The computation follows the equation:  M = C @ C.T / t'
        where:
        - C is the temporal components matrix (components × time)
        - t' is the current timestep
        - M is the resulting correlation matrix (components × components)
    """
    # Get temporal components C
    # components x time
    C = traces.array

    frame_idx = C[AXIS.frame_coord].max().item()

    # Compute M = C * C.T / t'
    M = C @ C.rename(AXIS.component_rename) / (frame_idx + 1)

    return CompStats.from_array(M)


def ingest_frame(component_stats: CompStats, frame: Frame, new_traces: PopSnap) -> CompStats:
    """
    Update component statistics using current frame and component.

        M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T

        where:
        - M_t is the component-wise sufficient statistics at time t
        - y_t is the current frame
        - c_t is the current temporal component
        - t is the current timestep

    Args:
        frame (Frame): Current frame y_t.
            Shape: (height × width)
        new_traces (Traces): Current temporal component c_t.
            Shape: (components)
    Return:
        component_stats (ComponentStats): Updated component-statistics.
    """
    if new_traces.array is None:
        return component_stats

    # Compute scaling factors
    frame_idx = frame.array.coords[AXIS.frame_coord].item()
    prev_scale = frame_idx / (frame_idx + 1)
    new_scale = 1 / (frame_idx + 1)

    # New frame traces
    c_t = new_traces.array

    # Update component-wise statistics M_t
    # M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T
    new_corr = c_t @ c_t.rename(AXIS.component_rename)
    component_stats.array = (
        prev_scale * component_stats.array + new_scale * new_corr
    ).reset_coords([AXIS.timestamp_coord, AXIS.frame_coord], drop=True)

    return component_stats


def ingest_component(component_stats: CompStats, traces: Traces, new_trace: Trace) -> CompStats:
    """
    Update component statistics with new components.

    Updates M_t according to the equation:
    M_t = (1/t) [ tM_t,         C_buf^T c_new  ]
             [ c_new C_buf^T, ||c_new||^2   ]
    where:
    - t is the current frame index
    - M_t is the existing component statistics
    - C_buf are the traces in the buffer
    - c_new are the new component traces

    Args:
        traces (Traces): Current temporal traces in buffer
        new_trace (Traces): Newly detected temporal components
    """
    if new_trace.array is None:
        return component_stats

    # Get current frame index (starting with 1)
    t = new_trace.array[AXIS.frame_coord].max().item() + 1

    c_new = new_trace.array.volumize.dim_with_coords(
        dim=AXIS.component_dim, coords=[AXIS.id_coord, AXIS.confidence_coord]
    )
    M = component_stats.array

    if M is None or M.size == 1:
        component_stats.array = initialize(traces).array
        return component_stats

    if c_new[AXIS.id_coord].item() in M[AXIS.id_coord].values:
        # trace REPLACEMENT
        dim_idx = np.where(M[AXIS.id_coord].values == c_new[AXIS.id_coord].item())[0].tolist()
        M = M.drop_sel({AXIS.component_dim: dim_idx, f"{AXIS.component_dim}'": dim_idx})

    # think i also have to remove the ID from c_buf,
    # since it's been already added in trace.component_ingest
    c_buf = traces.array
    id_idx = np.where(c_buf[AXIS.id_coord].values == c_new[AXIS.id_coord].item())[0].tolist()
    c_buf = c_buf.drop_sel({AXIS.component_dim: id_idx})

    # Compute cross-correlation between buffer and new components
    # C_buf^T c_new
    # C_buf has to be the same number of frames as c_new
    bottom_left_corr = c_buf @ c_new.rename(AXIS.component_rename) / t
    top_right_corr = c_buf.rename(AXIS.component_rename) @ c_new / t

    # Compute auto-correlation of new components
    # ||c_new||^2
    auto_corr = c_new @ c_new.rename(AXIS.component_rename) / t

    # Create the block matrix structure
    # Top block: [M_scaled, cross_corr]
    top_block = xr.concat([M, top_right_corr], dim=AXIS.component_dim)
    # Bottom block: [cross_corr.T, auto_corr]
    bottom_block = xr.concat([bottom_left_corr, auto_corr], dim=AXIS.component_dim)
    # Combine blocks
    component_stats.array = xr.concat([top_block, bottom_block], dim=f"{AXIS.component_dim}'")

    return component_stats
