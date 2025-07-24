import xarray as xr
from noob.node import Node

from cala.models import AXIS, CompStat, Frame, Traces


class ComponentStats(Node):
    component_stats_: CompStat = None

    def initialize(self, traces: Traces) -> CompStat:
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

        frame_idx = C.sizes[self.params.frames_dim]

        # Compute M = C * C.T / t'
        M = C @ C.rename({self.params.component_dim: f"{self.params.component_dim}'"}) / frame_idx

        self.component_stats_ = CompStat(array=M.assign_coords(C.coords))
        return self.component_stats_

    def ingest_frame(self, frame: Frame, traces: Traces) -> CompStat:
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
            traces (Traces): Current temporal component c_t.
                Shape: (components)
        Return:
            component_stats (ComponentStats): Updated component-statistics.
        """
        # Compute scaling factors
        frame_idx = frame.array.coords[AXIS.frame_coord].item()
        prev_scale = frame_idx / (frame_idx + 1)
        new_scale = 1 / (frame_idx + 1)

        # New frame traces
        c_t = traces.array.isel({AXIS.frames_dim: -1})

        # Update component-wise statistics M_t
        # M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T
        new_corr = c_t @ c_t.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})
        self.component_stats_.array = (
            prev_scale * self.component_stats_.array + new_scale * new_corr
        )

        return self.component_stats_

    def ingest_component(self, traces: Traces) -> CompStat: ...
