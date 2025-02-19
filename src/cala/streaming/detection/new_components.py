import numpy as np
from river.base import SupervisedTransformer

from cala.streaming.core.estimates import Estimates


class NewComponentDetector(SupervisedTransformer):
    """Detects new components in residuals"""

    def detect(self, residual_frame):
        pass

    def _add_new_components(
        self, new_components: dict, estimates: Estimates
    ) -> Estimates:
        """Add newly detected components to estimates

        Args:
            new_components: Dictionary with new spatial and temporal components
        """
        if estimates.spatial_footprints is None:
            estimates.spatial_components = new_components["spatial"]
            estimates.temporal_components = new_components["temporal"]
        else:
            estimates.spatial_footprints = np.hstack(
                [estimates.spatial_footprints, new_components["spatial"]]
            )
            estimates.temporal_traces = np.vstack(
                [estimates.temporal_traces, new_components["temporal"]]
            )
        return estimates
