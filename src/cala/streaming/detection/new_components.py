import numpy as np
from river.base import SupervisedTransformer


class NewComponentDetector(SupervisedTransformer):
    """Detects new components in residuals"""

    def detect(self, residual_frame):
        pass

    def _add_new_components(
            self,
            new_components: dict,
    ) -> np.ndarray:
        """Add newly detected components to estimates

        Args:
            new_components: Dictionary with new spatial and temporal components
        """
        return np.array([])
