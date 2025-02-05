from river.base import SupervisedTransformer
from cala.streaming_service.core.estimates import Estimates
import numpy as np


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
        if estimates.spatial_components is None:
            estimates.spatial_components = new_components["spatial"]
            estimates.temporal_components = new_components["temporal"]
        else:
            estimates.spatial_components = np.hstack(
                [estimates.spatial_components, new_components["spatial"]]
            )
            estimates.temporal_components = np.vstack(
                [estimates.temporal_components, new_components["temporal"]]
            )
        return estimates
